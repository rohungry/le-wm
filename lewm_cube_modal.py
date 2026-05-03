"""
Reproduce the LeWorldModel (LeWM) OGBench-Cube planning eval on Modal.

What this does
--------------
1. Builds a container image with Python 3.10, MuJoCo's EGL deps, the
   `stable-worldmodel[train,env]` package, and a clone of `lucas-maes/le-wm`.
2. Downloads the HF mirror checkpoint `quentinll/lewm-cube` (weights.pt +
   config.json), and converts it to the `_object.ckpt` format that
   `eval.py` / `swm.policy.AutoCostModel` expect.
3. Runs `python eval.py --config-name=cube.yaml policy=cube/lewm` on a
   single A10 GPU (Modal's closest cousin to an RTX 3090; both are 24 GB
   Ampere parts).

How to run (from your laptop, in a directory containing this file)
------------------------------------------------------------------
    pip install modal             # if not already
    modal token new               # one-time auth, if not already
    modal run lewm_cube_modal.py  # downloads + converts + runs eval

The first run takes ~5–10 min to build the image. Subsequent runs reuse
the cached image and the checkpoint volume.

For interactive debugging instead of a one-shot run:
    modal shell lewm_cube_modal.py::run_eval
"""

import modal

# -----------------------------------------------------------------------------
# Container image
# -----------------------------------------------------------------------------
# Python 3.10 to match the repo's `uv venv --python=3.10` recommendation.
# apt packages: git for cloning; the libGL/EGL/OSMesa stack so MuJoCo can
# render offscreen on a GPU container with no display.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "build-essential",
        "swig",  # needed by box2d-py (transitive via gym[box2d])
        "zstd",  # decompress cube_single_expert.tar.zst
        # MuJoCo / EGL headless rendering deps
        "libgl1",
        "libglib2.0-0",
        "libegl1",
        "libgles2",
        "libosmesa6",
        "libglu1-mesa",
        "libglvnd-dev",
        "patchelf",
        "ffmpeg",
    )
    .run_commands(
        # Clone the LeWM code (eval.py + config/eval/*.yaml live here).
        "git clone https://github.com/lucas-maes/le-wm.git /root/le-wm",
    )
    .pip_install(
        "huggingface_hub",
        "hydra-core",
    )
    .run_commands(
        # `gym==0.21.0` (pulled in transitively by `stable-baselines3==1.8.0`,
        # itself a dep of `stable-worldmodel[env]`) has a malformed setup.py
        # that fails to build under setuptools >= 66.  Pin old build tooling,
        # install gym with build isolation OFF so it picks them up, then let
        # the resolver install the rest normally.
        "pip install --upgrade 'pip<24' 'setuptools==65.5.0' 'wheel<0.40'",
        "pip install --no-build-isolation 'gym==0.21.0'",
        # `[env]` brings mujoco / ogbench / gymnasium; `[train]` brings
        # stable-pretraining (used to build the ViT encoder).
        "pip install 'stable-worldmodel[train,env]'",
    )
    .env(
        {
            "MUJOCO_GL": "egl",            # GPU-accelerated headless render
            "PYOPENGL_PLATFORM": "egl",
            "STABLEWM_HOME": "/cache",     # checkpoint + data root
            "WANDB_MODE": "offline",       # don't require a wandb account
            "HF_HOME": "/cache/hf",        # huggingface download cache
            "HYDRA_FULL_ERROR": "1",       # nicer tracebacks on config errors
        }
    )
    # Overlay the local module.py on top of the cloned repo so you can edit
    # the SIGReg class on your laptop and the next `modal run` picks it up
    # without an image rebuild. Requires a `module.py` next to this script
    # (typically: just symlink or copy your local clone's module.py here).
    .add_local_file("module.py", "/root/le-wm/module.py")
)

app = modal.App("lewm-cube", image=image)

# Persistent volume so we only download / convert the checkpoint once.
volume = modal.Volume.from_name("lewm-cube-cache", create_if_missing=True)
VOL_MNT = "/cache"

# A10 = 24 GB Ampere, the closest Modal SKU to an RTX 3090. L4 also works
# for inference and is ~30% cheaper if you want to swap.
GPU = "A10"

# Planning eval can be slow (CEM rollouts in MuJoCo over many seeds). 3h is
# generous; bump if cube.yaml runs many episodes.
EVAL_TIMEOUT = 60 * 60 * 3


# -----------------------------------------------------------------------------
# Step 1 — download HF checkpoint and convert weights.pt → lewm_object.ckpt
# -----------------------------------------------------------------------------
@app.function(
    gpu=None,                          # CPU is fine for download + conversion
    volumes={VOL_MNT: volume},
    timeout=60 * 30,
)
def download_and_convert():
    """Fetch quentinll/lewm-cube and serialize a ready-to-load LeWM module."""
    import json
    import sys
    from pathlib import Path

    import torch
    from huggingface_hub import snapshot_download

    # The HF config.json's `_target_` keys point at
    # `stable_worldmodel.wm.lewm.*`, but the released stable-worldmodel
    # (0.0.6) does not expose that submodule.  The README's "From the
    # Hugging Face mirror" snippet builds the model manually from the
    # cloned repo's jepa.py + module.py — do the same here.
    sys.path.insert(0, "/root/le-wm")
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP
    import stable_pretraining as spt

    cache = Path("/cache")
    src = cache / "hf_cube"
    src.mkdir(parents=True, exist_ok=True)

    # 1a. Download config.json + weights.pt from the HF mirror.
    snapshot_download(
        repo_id="quentinll/lewm-cube",
        local_dir=str(src),
        local_dir_use_symlinks=False,
    )

    cfg = json.loads((src / "config.json").read_text())

    # Hydra leaves `_target_` / `_partial_` markers in each sub-dict; strip
    # them so the remaining keys are valid Python kwargs.
    def kw(d):
        return {k: v for k, v in d.items() if not k.startswith("_")}

    # 1b. Build encoder / predictor / projector / action embedder by hand.
    encoder = spt.backbone.utils.vit_hf(
        cfg["encoder"]["size"],
        patch_size=cfg["encoder"]["patch_size"],
        image_size=cfg["encoder"]["image_size"],
        pretrained=False,
        use_mask_token=False,
    )

    def mlp(name):
        return MLP(
            input_dim=cfg[name]["input_dim"],
            output_dim=cfg[name]["output_dim"],
            hidden_dim=cfg[name]["hidden_dim"],
            norm_fn=torch.nn.BatchNorm1d,
        )

    model = JEPA(
        encoder=encoder,
        predictor=ARPredictor(**kw(cfg["predictor"])),
        action_encoder=Embedder(**kw(cfg["action_encoder"])),
        projector=mlp("projector"),
        pred_proj=mlp("pred_proj"),
    )

    # 1c. Load the state dict.  strict=False is defensive — print any
    # mismatch instead of crashing, so a small key drift (e.g. BN running
    # stats) is debuggable rather than fatal.
    sd = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[warn] missing keys: {len(missing)}, unexpected: {len(unexpected)}")
        if missing:
            print(" missing[:5]:", missing[:5])
        if unexpected:
            print(" unexpected[:5]:", unexpected[:5])
    model.eval()

    # 1d. Save where eval.py / AutoCostModel expect:
    #     $STABLEWM_HOME/cube/lewm_object.ckpt
    out = cache / "cube" / "lewm_object.ckpt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, out)
    volume.commit()
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


# -----------------------------------------------------------------------------
# Step 1.5 — download the OGBench-Cube dataset (~46 GB compressed)
# -----------------------------------------------------------------------------
# eval.py (via stable_worldmodel.data.HDF5Dataset) reads
# `$STABLEWM_HOME/ogbench/cube_single_expert.h5` to source episode start /
# goal observations during planning, even in inference mode. The archive
# is hosted on the *dataset* repo (note: same name as the model repo, but
# different namespace).
@app.function(
    gpu=None,
    volumes={VOL_MNT: volume},
    timeout=60 * 60 * 2,  # 46 GB download + extract; 2 h is plenty
)
def download_dataset():
    """Fetch and extract cube_single_expert.h5 from the HF dataset mirror."""
    import os
    import shutil
    import subprocess
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    target = Path("/cache/ogbench/cube_single_expert.h5")
    if target.exists():
        print(f"{target} already present ({target.stat().st_size / 1e9:.1f} GB), skipping.")
        return

    target.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading cube_single_expert.tar.zst (~46 GB) ...")
    archive = hf_hub_download(
        repo_id="quentinll/lewm-cube",
        repo_type="dataset",
        filename="cube_single_expert.tar.zst",
        local_dir="/cache/hf_cube_data",
        local_dir_use_symlinks=False,
    )
    archive = Path(archive)
    print(f"Got {archive} ({archive.stat().st_size / 1e9:.1f} GB)")

    # Pipe through zstd -d for portability (some `tar` builds lack --zstd).
    print(f"Extracting to {target.parent} ...")
    subprocess.run(
        f"zstd -d -c {archive} | tar -x -C {target.parent}",
        shell=True,
        check=True,
    )

    # Archive layout isn't documented; if extraction landed elsewhere,
    # locate the .h5 file and move it to the expected path.
    if not target.exists():
        found = list(Path("/cache").rglob("cube_single_expert.h5"))
        if not found:
            raise FileNotFoundError("cube_single_expert.h5 not found after extraction")
        print(f"Moving {found[0]} -> {target}")
        shutil.move(str(found[0]), str(target))

    # Reclaim ~46 GB from the Volume.
    archive.unlink(missing_ok=True)

    volume.commit()
    print(f"Ready: {target} ({target.stat().st_size / 1e9:.1f} GB)")


# -----------------------------------------------------------------------------
# Step 2 — run the planning evaluation on OGBench-Cube
# -----------------------------------------------------------------------------
@app.function(
    gpu=GPU,
    volumes={VOL_MNT: volume},
    timeout=EVAL_TIMEOUT,
)
def run_eval():
    """Invoke `python eval.py --config-name=cube.yaml policy=cube/lewm`."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    ckpt = Path("/cache/cube/lewm_object.ckpt")
    if not ckpt.exists():
        raise FileNotFoundError(
            f"{ckpt} missing — run download_and_convert first "
            f"(`modal run lewm_cube_modal.py::download_and_convert`)."
        )

    # Smoke test: GPU is visible and torch can use it.
    import torch
    print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))

    # eval.py is a Hydra script — config_path is relative to its own dir,
    # so we cwd into the cloned repo.
    repo = "/root/le-wm"
    cmd = [
        sys.executable, "eval.py",
        "--config-name=cube.yaml",
        "policy=cube/lewm",
    ]
    print(">>", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=repo, env=os.environ.copy())

    # Persist any artifacts eval.py wrote under $STABLEWM_HOME (videos, json
    # results, etc.) so you can pull them with `modal volume get`.
    volume.commit()
    if proc.returncode != 0:
        raise RuntimeError(f"eval.py exited with code {proc.returncode}")
    print("eval.py completed successfully.")


# -----------------------------------------------------------------------------
# Helper — inspect the cube H5 file to figure out how to slice it
# -----------------------------------------------------------------------------
# The paper uses 10K episodes * 200 steps = 2M frames. The HF dataset is
# ~100 GB extracted, which is likely a much bigger superset. To match the
# paper's training recipe (and finish in ~2-3h instead of ~12h) we need to
# slice the H5 down. Run this once, paste the output, then we'll write a
# correct slicer in one shot rather than guessing at the layout.
@app.function(
    gpu=None,
    volumes={VOL_MNT: volume},
    timeout=60 * 10,
)
def inspect_dataset():
    """Print the structure of cube_single_expert.h5."""
    from pathlib import Path
    import h5py
    import numpy as np

    h5_path = Path("/cache/ogbench/cube_single_expert.h5")
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    print(f"File: {h5_path}  ({h5_path.stat().st_size / 1e9:.1f} GB on disk)")
    print("=" * 70)

    with h5py.File(h5_path, "r") as f:

        def walk(name, obj):
            indent = "  " * name.count("/")
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}- {name}  shape={obj.shape}  dtype={obj.dtype}")
            else:
                print(f"{indent}+ {name}/  (group)")

        print("Hierarchy:")
        f.visititems(walk)

        # Heuristics: many OGBench-style HDF5 datasets store flat
        # (total_frames, ...) arrays plus a `terminals` / `episode_ends`
        # array marking episode boundaries. Probe the common names.
        print("\nTop-level keys:", list(f.keys()))

        for k in f.keys():
            obj = f[k]
            if isinstance(obj, h5py.Dataset):
                print(f"\n[{k}] shape={obj.shape} dtype={obj.dtype}")
                if obj.ndim >= 1 and obj.shape[0] > 0:
                    head = obj[: min(5, obj.shape[0])]
                    print(f"  head: {np.asarray(head).flatten()[:10]}")
                if k in ("terminals", "dones", "episode_ends", "valids"):
                    arr = obj[:]
                    n_eps = int(np.sum(arr)) if arr.dtype != bool else int(arr.sum())
                    print(f"  -> implies ~{n_eps} episodes")

    print("\nDone. Paste this output back to Claude to design the slice.")


# -----------------------------------------------------------------------------
# Step 3 — train LeWM on OGBench-Cube from scratch
# -----------------------------------------------------------------------------
# train.py is `@hydra.main(config_path="./config/train", config_name="lewm")`,
# and the README's pattern is `python train.py data=pusht`. Cube's data
# config is at config/train/data/ogb.yaml, so the override is `data=ogb`.
#
# We override:
#   * data=ogb                     -- pick the cube dataset config
#   * wandb.enabled=false          -- skip wandb auth
#   * subdir=cube_trained          -- land checkpoints in /cache/cube_trained/
#                                     so the HF checkpoint at /cache/cube/ is
#                                     untouched (handy for A/B comparisons).
#   * trainer.max_epochs=10        -- match paper
#
# After training, copy the resulting `*_object.ckpt` to the path eval.py
# expects (/cache/cube/lewm_object.ckpt).
# train.py is `@hydra.main(config_path="./config/train", config_name="lewm")`,
# and the README's pattern is `python train.py data=pusht`. Cube's data
# config is at config/train/data/ogb.yaml, so the override is `data=ogb`.
#
# We override:
#   * data=ogb                     -- pick the cube dataset config
#   * wandb.enabled=false          -- skip wandb auth
#   * subdir=cube_trained          -- land checkpoints in /cache/cube_trained/
#                                     so the HF checkpoint at /cache/cube/ is
#                                     untouched (handy for A/B comparisons).
#
# After training, copy the resulting `*_object.ckpt` to the path eval.py
# expects (/cache/cube/lewm_object.ckpt).
@app.function(
    gpu="L40S",
    volumes={VOL_MNT: volume},
    timeout=60 * 60 * 16,  # 10 epochs at bs 128 ≈ 12 h on L40S; 16 h margin
)
def train_cube():
    """Run train.py with cube dataset and copy the result into eval's path."""
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    if not Path("/cache/ogbench/cube_single_expert.h5").exists():
        raise FileNotFoundError(
            "Dataset missing. Run `download_dataset` first or use the "
            "train_and_eval entrypoint which chains them together."
        )

    import torch
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))

    cmd = [
        sys.executable, "train.py",
        "data=ogb",
        "wandb.enabled=false",
        "subdir=cube_trained",
        "trainer.max_epochs=10",   # paper trains for 10 epochs
        # batch_size stays at the lewm.yaml default of 128 — paper specifies
        # 128 with sub-trajectories of size 4 (4 frames * 5-action blocks).
    ]
    print(">>", " ".join(cmd))
    proc = subprocess.run(cmd, cwd="/root/le-wm", env=os.environ.copy())
    if proc.returncode != 0:
        raise RuntimeError(f"train.py exited with code {proc.returncode}")

    # Locate the *_object.ckpt the callback wrote (suffix may vary; glob it).
    candidates = sorted(
        Path("/cache/cube_trained").glob("*_object.ckpt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        # Fall back to anything with "object" in the name.
        candidates = sorted(
            Path("/cache/cube_trained").glob("*object*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    if not candidates:
        raise FileNotFoundError(
            "No *_object.ckpt found under /cache/cube_trained — inspect with "
            "`modal volume ls lewm-cube-cache cube_trained` and adjust the glob."
        )

    src = candidates[0]
    dst = Path("/cache/cube/lewm_object.ckpt")
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Copying trained checkpoint {src} -> {dst}")
    shutil.copy2(src, dst)

    volume.commit()
    print(f"Trained checkpoint ready at {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


# -----------------------------------------------------------------------------
# Tier-0 smoke test (~2 min) — fastest possible "did anything break"
# -----------------------------------------------------------------------------
# Runs train.py for one mini-epoch (20 batches) with no wandb. Use this
# right after editing module.py to catch shape errors, NaNs, or import
# failures before paying for a longer run.
@app.function(
    gpu="L40S",
    volumes={VOL_MNT: volume},
    timeout=60 * 30,
)
def smoke_test():
    import os, subprocess, sys
    if not Path_exists("/cache/ogbench/cube_single_expert.h5"):
        raise FileNotFoundError("Run download_dataset first.")
    cmd = [
        sys.executable, "train.py",
        "data=ogb",
        "wandb.enabled=false",
        "subdir=smoke_test",
        "trainer.max_epochs=1",
        "+trainer.limit_train_batches=20",
        "+trainer.limit_val_batches=5",
    ]
    print(">>", " ".join(cmd))
    proc = subprocess.run(cmd, cwd="/root/le-wm", env=os.environ.copy())
    if proc.returncode != 0:
        raise RuntimeError(f"smoke_test exited with code {proc.returncode}")
    print("Smoke test OK — your module.py edits don't crash the loop.")


# -----------------------------------------------------------------------------
# Tier-2 short / Tier-3 full variant training (parameterized)
# -----------------------------------------------------------------------------
# Each variant lands in its own subdir, so baseline and variants coexist.
# Pass any extra Hydra overrides as a single space-separated string,
# e.g. --overrides "loss.sigreg.kwargs.variant=pca loss.sigreg.kwargs.reduce_dim=64".
@app.function(
    gpu="L40S",
    volumes={VOL_MNT: volume},
    timeout=60 * 60 * 12,
)
def train_variant(variant_name: str, max_epochs: int = 100, overrides: str = ""):
    import os, shutil, subprocess, sys
    from pathlib import Path

    if not Path("/cache/ogbench/cube_single_expert.h5").exists():
        raise FileNotFoundError("Run download_dataset first.")

    extra = overrides.split() if overrides else []
    cmd = [
        sys.executable, "train.py",
        "data=ogb",
        "wandb.enabled=false",
        f"subdir={variant_name}",
        f"trainer.max_epochs={max_epochs}",
        *extra,
    ]
    print(">>", " ".join(cmd))
    proc = subprocess.run(cmd, cwd="/root/le-wm", env=os.environ.copy())
    if proc.returncode != 0:
        raise RuntimeError(f"train.py exited with code {proc.returncode}")

    # Locate the *_object.ckpt and stage a copy under /cache/variants/<name>/.
    run_dir = Path("/cache") / variant_name
    candidates = sorted(
        list(run_dir.glob("*_object.ckpt")) + list(run_dir.glob("*object*")),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {run_dir}")
    final = Path("/cache/variants") / variant_name / "lewm_object.ckpt"
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidates[0], final)
    volume.commit()
    print(f"Trained '{variant_name}' -> {final} ({final.stat().st_size / 1e6:.1f} MB)")


# -----------------------------------------------------------------------------
# Eval a specific variant by swapping its checkpoint into the path eval.py
# expects, then invoking the existing eval pipeline.
# -----------------------------------------------------------------------------
@app.function(
    gpu="L40S",
    volumes={VOL_MNT: volume},
    timeout=EVAL_TIMEOUT,
)
def eval_variant(variant_name: str):
    import os, shutil, subprocess, sys
    from pathlib import Path

    src = Path("/cache/variants") / variant_name / "lewm_object.ckpt"
    if not src.exists():
        raise FileNotFoundError(
            f"No staged checkpoint for '{variant_name}' at {src}. "
            f"Train it first with `modal run lewm_cube_modal.py::train_variant "
            f"--variant-name {variant_name}`."
        )
    dst = Path("/cache/cube/lewm_object.ckpt")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"Loaded {variant_name} checkpoint into {dst}")

    cmd = [sys.executable, "eval.py", "--config-name=cube.yaml", "policy=cube/lewm"]
    print(">>", " ".join(cmd))
    proc = subprocess.run(cmd, cwd="/root/le-wm", env=os.environ.copy())
    volume.commit()
    if proc.returncode != 0:
        raise RuntimeError(f"eval.py exited with code {proc.returncode}")
    print(f"Eval done for variant '{variant_name}'.")


# Helper used by smoke_test (avoids importing pathlib at module top-level).
def Path_exists(p):
    from pathlib import Path
    return Path(p).exists()


# -----------------------------------------------------------------------------
# Entrypoints
# -----------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    """Inference-only with the HF checkpoint."""
    download_and_convert.remote()
    download_dataset.remote()
    run_eval.remote()


@app.local_entrypoint()
def train_and_eval():
    """Train cube from scratch (single combined entrypoint, baseline-style)."""
    download_dataset.remote()
    train_cube.remote()
    run_eval.remote()
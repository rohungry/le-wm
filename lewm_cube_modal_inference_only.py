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
    sys.exit(proc.returncode)


# -----------------------------------------------------------------------------
# One-shot entrypoint: `modal run lewm_cube_modal.py`
# -----------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    download_and_convert.remote()
    download_dataset.remote()
    run_eval.remote()
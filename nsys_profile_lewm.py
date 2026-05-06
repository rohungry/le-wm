"""
Profile a short LeWM cube training run with torch.profiler.

Why torch.profiler instead of nsys
----------------------------------
Modal's L40S containers run with a virtualized NVML / GPU-UUID layer that
nsys (both 2024.5 and 2026.1) cannot reconcile — the daemon throws either
`ConvertGpuTicksToSyncNs` or `No GPU associated to the given UUID` at
report-finalization time and writes nothing to disk. torch.profiler runs
entirely in-process and uses libcupti directly, sidestepping the daemon
that's failing.

What you get
------------
* /cache/profiles/lewm_trace.json  — Chrome trace (open in
                                     https://ui.perfetto.dev OR
                                     chrome://tracing OR
                                     edge://tracing).  Shows per-kernel
                                     CUDA timeline, memcpy events, CPU
                                     stacks during GPU idle.
* /cache/profiles/lewm_summary.txt — top-30 ops by self GPU time and by
                                     self CPU time. Quick scan to find
                                     the bottlenecks.

How to run
----------
    modal run torch_profile_lewm.py

After it finishes, pull the artifacts:
    modal volume get lewm-cube-cache profiles/lewm_trace.json   ./lewm_trace.json
    modal volume get lewm-cube-cache profiles/lewm_summary.txt  ./lewm_summary.txt

Drop lewm_trace.json onto https://ui.perfetto.dev — no install required.

What the profiler captures
--------------------------
* CPU activity (Python ops, autograd nodes) on its own track
* CUDA kernel launches with durations on a per-stream track
* cudaMemcpy events (host↔device, device↔device) with sizes
* Operator-level rollup so each forward/backward/optimizer-step is grouped

Does NOT capture L1/L2/SRAM cache hit rates — those need ncu (Nsight
Compute), which has the same virtualization issue as nsys on Modal. For
data-movement understanding short of cache-line analysis, this is the
right tool.
"""

import modal

# Reuse the same image recipe as the training script. We don't actually
# need nsys here, but keeping the build steps identical means Modal's
# build cache is shared and the image rebuild is fast (or zero) if you've
# already built nsys_profile_lewm.py.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "build-essential",
        "swig",
        "zstd",
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
        "git clone https://github.com/lucas-maes/le-wm.git /root/le-wm",
    )
    .pip_install("huggingface_hub", "hydra-core")
    .run_commands(
        "pip install --upgrade 'pip<24' 'setuptools==65.5.0' 'wheel<0.40'",
        "pip install --no-build-isolation 'gym==0.21.0'",
        "pip install 'stable-worldmodel[train,env]'",
    )
    .env(
        {
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "STABLEWM_HOME": "/cache",
            "WANDB_MODE": "offline",
            "HF_HOME": "/cache/hf",
            "HYDRA_FULL_ERROR": "1",
        }
    )
)

app = modal.App("lewm-cube-torch-profile", image=image)

volume = modal.Volume.from_name("lewm-cube-cache", create_if_missing=False)
VOL_MNT = "/cache"


@app.function(
    gpu="L40S",
    volumes={VOL_MNT: volume},
    timeout=60 * 30,
)
def profile_train():
    """Build the LeWM model + dataloader and profile a few training steps.

    We don't run train.py as a subprocess. Instead we import its
    components and run a manual training loop inside the profiler
    context so the profiler boundaries are exactly the steps we want.
    """
    import sys
    from functools import partial
    from pathlib import Path

    import torch
    from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

    if not Path("/cache/ogbench/cube_single_expert.h5").exists():
        raise FileNotFoundError(
            "Dataset missing at /cache/ogbench/cube_single_expert.h5"
        )

    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
    print("device:", torch.cuda.get_device_name(0))

    out_dir = Path("/cache/profiles")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Make le-wm importable so we can grab JEPA / ARPredictor / etc.
    sys.path.insert(0, "/root/le-wm")

    # Build the Hydra config the same way train.py does, but using
    # Compose API so we don't have to fork a subprocess.
    from hydra import initialize_config_dir, compose
    from omegaconf import open_dict

    with initialize_config_dir(config_dir="/root/le-wm/config/train", version_base=None):
        cfg = compose(
            config_name="lewm",
            overrides=["data=ogb"],
        )

    # === Replicate train.py's setup, minus Lightning ===
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP, SIGReg
    from utils import get_column_normalizer, get_img_preprocessor

    print("\n[1/5] Building dataset ...")
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    ]
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))
    dataset.transform = spt.data.transforms.Compose(*transforms)

    print(f"  dataset has {len(dataset)} samples")

    # Use fewer workers than the full training run — we only need a few
    # batches and we don't want dataloader spin-up to dominate the
    # profile.
    loader_cfg = dict(cfg.loader)
    loader_cfg["num_workers"] = 2
    loader_cfg["persistent_workers"] = False
    loader_cfg.pop("prefetch_factor", None)
    loader = torch.utils.data.DataLoader(
        dataset, **loader_cfg, shuffle=True, drop_last=True,
    )

    print("\n[2/5] Building model ...")
    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )
    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim,
                    hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    pred_proj = MLP(input_dim=hidden_dim, output_dim=embed_dim,
                    hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    world_model = JEPA(
        encoder=encoder, predictor=predictor, action_encoder=action_encoder,
        projector=projector, pred_proj=pred_proj,
    ).cuda()
    sigreg = SIGReg(**cfg.loss.sigreg.kwargs).cuda()

    optimizer = torch.optim.AdamW(
        world_model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )

    print(f"  model has {sum(p.numel() for p in world_model.parameters()) / 1e6:.1f}M params")

    # === Training step closure (mirrors train.py's lejepa_forward) ===
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    def step(batch):
        # Move to GPU and run forward/backward/step. The profiler will
        # tag each phase via record_function for readability in the
        # Chrome trace.
        batch = {
            k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)

        with torch.profiler.record_function("encode"):
            out = world_model.encode(batch)
        emb = out["emb"]
        act_emb = out["act_emb"]
        with torch.profiler.record_function("predict"):
            pred_emb = world_model.predict(emb[:, :ctx_len], act_emb[:, :ctx_len])
        tgt_emb = emb[:, n_preds:]

        with torch.profiler.record_function("loss"):
            pred_loss = (pred_emb - tgt_emb).pow(2).mean()
            sigreg_loss = sigreg(emb.transpose(0, 1))
            loss = pred_loss + lambd * sigreg_loss

        with torch.profiler.record_function("backward"):
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

        with torch.profiler.record_function("optimizer_step"):
            optimizer.step()

        return loss.item()

    # === Run with the profiler ===
    # Schedule:
    #   wait=2     skip the first 2 steps (Python init, first-batch
    #              caching, autotuner warmup)
    #   warmup=3   warm up the profiler itself (cuDNN benchmark, etc)
    #   active=10  collect 10 steady-state steps
    #   repeat=1   one cycle, then stop
    print("\n[3/5] Running profiled training loop ...")
    prof_schedule = schedule(wait=2, warmup=3, active=10, repeat=1)

    def trace_handler(prof):
        tr = out_dir / "lewm_trace.json"
        prof.export_chrome_trace(str(tr))
        print(f"\n  wrote {tr} ({tr.stat().st_size / 1e6:.1f} MB)")

        summary = out_dir / "lewm_summary.txt"
        with open(summary, "w") as f:
            f.write("=== Top 30 ops by self CUDA time ===\n\n")
            f.write(prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=30,
            ))
            f.write("\n\n=== Top 30 ops by self CPU time ===\n\n")
            f.write(prof.key_averages().table(
                sort_by="self_cpu_time_total", row_limit=30,
            ))
        print(f"  wrote {summary} ({summary.stat().st_size / 1e3:.0f} KB)")

    total_steps = 2 + 3 + 10  # wait + warmup + active
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule,
        on_trace_ready=trace_handler,
        record_shapes=True,        # tag tensor shapes on each op
        profile_memory=False,      # keep trace size manageable
        with_stack=False,          # stacks blow up the trace; disable
    ) as prof:
        world_model.train()
        loader_iter = iter(loader)
        for i in range(total_steps):
            batch = next(loader_iter)
            loss = step(batch)
            prof.step()
            print(f"  step {i + 1}/{total_steps}  loss={loss:.4f}")

    print("\n[4/5] Profile complete.")

    # The summary file was already written by trace_handler when the
    # active window closed. Nothing more to do here on the analysis
    # side; the user pulls lewm_summary.txt to see the top-N ops.

    volume.commit()

    print(
        "\nDone. Pull to your laptop with:\n"
        "  modal volume get lewm-cube-cache profiles/lewm_trace.json  ./lewm_trace.json\n"
        "  modal volume get lewm-cube-cache profiles/lewm_summary.txt ./lewm_summary.txt\n"
        "\n"
        "Open the trace at https://ui.perfetto.dev (drag-and-drop the .json)."
    )


@app.function(
    gpu=None,                          # CPU only — just for browsing files
    volumes={VOL_MNT: volume},
    timeout=60 * 60,
)
def inspect_profiles():
    """Shell entry point for browsing /cache/profiles/ interactively.

    Use:  modal shell torch_profile_lewm.py::inspect_profiles

    From inside the shell:
        ls -lh /cache/profiles/                            # see artifacts
        less /cache/profiles/lewm_summary.txt              # top-N ops table
        head -c 2000 /cache/profiles/lewm_trace.json | python -m json.tool
                                                            # peek at trace structure
    """
    print("Files in /cache/profiles/:")
    import os
    for f in sorted(os.listdir("/cache/profiles")):
        path = f"/cache/profiles/{f}"
        size = os.path.getsize(path)
        if size > 1e6:
            print(f"  {f:30s} {size / 1e6:7.1f} MB")
        else:
            print(f"  {f:30s} {size / 1e3:7.1f} KB")


@app.local_entrypoint()
def main():
    profile_train.remote()
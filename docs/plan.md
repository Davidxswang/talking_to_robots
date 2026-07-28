# Plan: talking_to_robots

Goal: a fun, community-meaningful robot-learning project built on LeRobot.

## The 3-phase arc

1. **Phase 1 — LeRobot end-to-end in sim.** Train a policy on an existing
   benchmark task (PushT), evaluate it in the gym env, visualize dataset and
   rollouts. Deliverable: a trained policy with a measured success rate.
2. **Phase 2 — our own MuJoCo task.** Design a new task (MJCF model + gym env
   following the `gym-pusht` / `gym-aloha` package pattern), record or script
   demonstrations into `LeRobotDataset` format, train on it.
3. **Phase 3 — upstream.** Polish the task/env, fix anything we hit along the
   way, and open PRs against `huggingface/lerobot` (and/or publish the env +
   dataset + checkpoint on the HF Hub).

## Environment (set up 2026-07-27)

- Machine: RTX 4090 24 GB (driver 595.84, CUDA 13.2), Python 3.13.11 via uv.
- Pinned in `uv.lock`: lerobot 0.6.0 (PyPI) with extras
  `[pusht,diffusion,smolvla,training,evaluation,dataset-viz]`, plus `mujoco`
  (explicit dep, for Phase 2).
- Key versions: torch 2.11.0+cu130, mujoco 3.11.0, gymnasium 1.3.0,
  gym-pusht 0.1.6, wandb 0.27.2, diffusers 0.35.2, transformers 5.5.4,
  rerun-sdk 0.33.1. CUDA verified available.
- Training entrypoints (lerobot 0.6.0 console scripts, flags verified against
  `--help`): `lerobot-train`, `lerobot-eval`, `lerobot-dataset-viz`.
- Dataset: `lerobot/pusht` — 206 episodes, 25,650 frames, 10 fps; features
  `observation.image` (96,96,3) video, `observation.state` (2,), `action` (2,).
  ~7.5 MB, cached in the default HF cache
  (`~/.cache/huggingface/lerobot/hub/datasets--lerobot--pusht/`). Small enough
  to stay in the HF cache; no copy under `/data` needed.
- Artifacts: everything under central `/outputs/talking_to_robots/` (repo
  `outputs/` symlinks there).
- Smoke test passed 2026-07-27 (CPU only): `uv run python scripts/smoke_test.py`
  loads the dataset, builds the 262.7M-param Diffusion Policy via
  `make_policy`, runs one forward pass (loss ≈ 1.02).

## Phase-1 runbook

### Run 1: Diffusion Policy on PushT

Reference reproduction (HF model card `lerobot/diffusion_pusht`): batch 64,
200k steps, env-eval every 25k; final success rate 65.4% over 500 episodes
(original Diffusion Policy paper: 64.2%).

**Step 0 — 10-min probe (do this first).** The LeRobot hardware guide
(hf.co/docs/lerobot/main/en/hardware_guide) gives diffusion ~8–14 GB VRAM at
batch 8 on 640×480 images; PushT is 96×96 so both VRAM and step time will be
much lower, but the guide has no PushT number → wall-clock is **unknown,
measure on a probe first**:

```bash
cd ~/projects/talking_to_robots
RUN=diffusion_pusht_probe_$(date +%Y%m%d_%H%M)
mkdir -p /outputs/talking_to_robots/runs
uv run lerobot-train \
  --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/pusht --env.type=pusht \
  --eval.use_async_envs=false \
  --batch_size=64 --steps=1000 --log_freq=50 \
  --env_eval_freq=1000 --save_freq=1000 \
  --output_dir=/outputs/talking_to_robots/runs/$RUN \
  --job_name=$RUN --wandb.enable=false \
  2>&1 | tee /outputs/talking_to_robots/runs/$RUN.log
```

**Probe result (2026-07-28, RTX 4090, run `diffusion_pusht_probe_20260728_0009`):**
11.1 steps/s steady, 4.97 GB VRAM (`mem_gb` from train log), loss 1.0 → 0.039
over 1k steps; 50-episode in-training eval works (0% success at 1k steps, as
expected — reference needs 200k). Extrapolation: 200k steps ≈ 5.0 h train
+ ~1 min/eval × 8 evals ≈ **5–5.5 h wall-clock** → full 200k run is fine, no
fallback needed.

Two flags above were learned the hard way (both crash otherwise, lerobot 0.6.0):
- `--policy.push_to_hub=false` — push-to-hub defaults ON and `cfg.validate()`
  hard-fails without a `--policy.repo_id`.
- `--eval.use_async_envs=false` — async vector-env workers raise
  `NamespaceNotFound: gym_pusht` (worker subprocesses never import the package
  that registers the env); sync envs are fine for cheap 2D PushT.
- Also: `tee` masks the train command's exit code — prefix with
  `set -o pipefail` when scripting these.

Watch `nvidia-smi` (expect comfortable headroom on 24 GB) and read steps/s from
the log; extrapolate: `200_000 / steps_per_s / 3600` hours. If the full 200k
run would exceed ~24 h, drop to `--steps=100000` (the lerobot default) and add
`--policy.scheduler_decay_steps=100000` so the LR schedule matches (hardware
guide: shorten the schedule when you shorten training).

**Full run** (after the probe):

```bash
cd ~/projects/talking_to_robots
RUN=diffusion_pusht_$(date +%Y%m%d_%H%M)
uv run lerobot-train \
  --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/pusht --env.type=pusht \
  --eval.use_async_envs=false \
  --batch_size=64 --steps=200000 \
  --env_eval_freq=25000 --save_freq=25000 --seed=100000 \
  --output_dir=/outputs/talking_to_robots/runs/$RUN \
  --job_name=$RUN --wandb.enable=true \
  2>&1 | tee /outputs/talking_to_robots/runs/$RUN.log
```

Notes:
- wandb login already present in `~/.netrc`; `--wandb.enable=true` verified in
  the 0.6.0 CLI. In-env eval during training comes from `--env.type=pusht` +
  `--env_eval_freq` (flag renamed from the model card's old `--eval_freq`).
- Checkpoints land in `$RUN/checkpoints/<step>/pretrained_model` and
  `checkpoints/last/pretrained_model` (per LeRobot il_sim docs).
- `--output_dir` must be a fresh directory (timestamped names guarantee this).

### Eval + visualization (after training)

```bash
# 50-episode eval of the final checkpoint (writes eval videos to output_dir)
uv run lerobot-eval \
  --policy.path=/outputs/talking_to_robots/runs/$RUN/checkpoints/last/pretrained_model \
  --env.type=pusht --eval.n_episodes=50 --eval.batch_size=10 \
  --output_dir=/outputs/talking_to_robots/eval/$RUN \
  2>&1 | tee /outputs/talking_to_robots/eval/$RUN.log

# dataset visualization (rerun viewer)
uv run lerobot-dataset-viz --repo-id lerobot/pusht --episode-index 0
```

Success bar: reference checkpoint reports 65.4% success / 0.955 max-overlap
over 500 episodes; anywhere near that on 50 episodes = phase-1 win.

### Run 2 (later this week): SmolVLA fine-tune

Language-conditioned run. Per the SmolVLA docs page
(hf.co/docs/lerobot/main/en/smolvla):
- Base checkpoint: `lerobot/smolvla_base` (450M params; `smolvla` extra
  already installed).
- Fine-tune command shape: `lerobot-train --policy.path=lerobot/smolvla_base
  --dataset.repo_id=<dataset> --batch_size=64 --steps=20000 ...`.
- Docs: 20k steps ≈ 4 h on an A100; hardware guide: smolvla ~10–16 GB VRAM →
  fits the 4090, expect slower than A100 (measure on a probe).
- Dataset requirement: LeRobotDataset with a natural-language task string.
  `lerobot/pusht` carries one ("Push the T-shaped block onto the T-shaped
  target.") but is single-task/no-camera-variety; pick a richer dataset (e.g.
  `lerobot/svla_so100_pickplace` from the SmolVLA paper) — decide with owner.

## Rules of the road

- All run artifacts → `/outputs/talking_to_robots/` (never repo-local).
- Per-run `.log` file via `tee` for every long job (see commands above).
- No GPU launches while owner sleeps; queue for morning go-ahead.
- Don't saturate the machine while the owner is actively using it (it's their
  daily-driver desktop) — modest footprint by default, full-throttle only on
  dedicated unattended runs with go-ahead.
- Feature branches + PRs after the initial scaffold commit; owner merges.
- Remote: `github.com/Davidxswang/talking_to_robots` (public, since 2026-07-27).

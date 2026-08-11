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
  `outputs/` symlinks there). **One self-contained folder per run**
  (layout adopted 2026-08-09; older artifacts migrated in place):

  ```
  /outputs/talking_to_robots/<run_name>/
    train.log   # tee'd lerobot-train output
    train/      # lerobot-train --output_dir (checkpoints/, wandb/, in-training eval/)
    eval.log    # tee'd lerobot-eval output (final eval)
    eval/       # lerobot-eval --output_dir (eval_info.json, videos/)
  ```

  This is the shape for new runs; migrated pre-2026-08-09 runs carry only the
  pieces they produced (e.g. crashed probes have just `train.log`, the dry run
  just `train/`).

  Why `train/` is a subfolder: lerobot-train refuses to start if its
  `--output_dir` already exists as a directory (unless `--resume=true`;
  `TrainPipelineConfig.validate()`), and the run folder must exist before
  `tee` can write `train.log` into it — so the trainer gets a
  not-yet-existing subdir inside the run folder. `lerobot-eval` has no such
  check — which cuts both ways: it will silently overwrite an existing
  `eval/` (see the eval block's warning).
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
RUN_DIR=/outputs/talking_to_robots/$RUN
mkdir -p $RUN_DIR
uv run lerobot-train \
  --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/pusht --env.type=pusht \
  --eval.use_async_envs=false \
  --batch_size=64 --steps=1000 --log_freq=50 \
  --env_eval_freq=1000 --save_freq=1000 \
  --output_dir=$RUN_DIR/train \
  --job_name=$RUN --wandb.enable=false \
  2>&1 | tee $RUN_DIR/train.log
```

**Probe result (2026-07-28, RTX 4090, run `diffusion_pusht_probe_20260728_0009`):**
11.1 steps/s steady, 4.97 GB VRAM steady-state (`mem_gb` from train log;
transient peak 5.29 GB), logged loss 0.88 (step 50) → 0.038 (step 950) at
log_freq=50; 50-episode in-training eval works (0% success at 1k steps, as
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
RUN_DIR=/outputs/talking_to_robots/$RUN
mkdir -p $RUN_DIR
uv run lerobot-train \
  --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/pusht --env.type=pusht \
  --eval.use_async_envs=false \
  --batch_size=64 --steps=200000 \
  --env_eval_freq=25000 --save_freq=25000 --seed=100000 \
  --output_dir=$RUN_DIR/train \
  --job_name=$RUN --wandb.enable=true \
  2>&1 | tee $RUN_DIR/train.log
```

Notes:
- wandb login already present in `~/.netrc`; `--wandb.enable=true` verified in
  the 0.6.0 CLI. In-env eval during training comes from `--env.type=pusht` +
  `--env_eval_freq` (flag renamed from the model card's old `--eval_freq`).
- Checkpoints land in `$RUN_DIR/train/checkpoints/<step>/pretrained_model` and
  `checkpoints/last/pretrained_model` (per LeRobot il_sim docs).
- `--output_dir` must be a fresh (not yet existing) directory — hence the
  `train/` subdir inside the pre-created run folder (see layout above).

**Run 1 result (2026-07-28, run `diffusion_pusht_20260728_0823`):** trained
clean, 200k steps in 4h15m (13.0 steps/s, ~5.0 GB VRAM per the log's
`mem_gb`; ~6.2 GB process total per nvidia-smi), wandb
`snowpine007/lerobot/runs/4eqo868n`. In-training 50-ep evals: 20 → 24 → 42 →
44 → 38 → 34 → 48 → 44% (plateau ~40–48% from 75k). Final 500-episode eval
(`/outputs/talking_to_robots/diffusion_pusht_20260728_0823/eval`):
**41.6% success / 0.858 avg max reward** vs reference 65.4% / 0.955 — a real
gap, not eval noise.

**Root cause (config diff vs `lerobot/diffusion_pusht` train_config.json on
the Hub):** lerobot 0.6.0's `--policy.type=diffusion` defaults drifted from
the published PushT recipe. Substantive differences (ref → ours):

| field | reference | 0.6.0 default (run 1) |
|---|---|---|
| `policy.horizon` | 16 | 64 |
| `policy.n_action_steps` | 8 | 32 |
| `policy.crop_shape` | [84, 84] random crop | None |
| `policy.use_group_norm` | true | false |
| `policy.pretrained_backbone_weights` | None (from scratch) | ImageNet ResNet18 |
| `policy.use_separate_rgb_encoder_per_camera` | false | true (1 camera → minor) |

The horizon/action-steps drift is the prime suspect: 32-step open-loop chunks
give far less closed-loop correction than the paper's 8, matching the observed
"gets close but can't finish" failures (avg max reward 0.86).

Intuition: at each observation the policy predicts a whole *chunk* of future
actions, then executes `n_action_steps` of them *open-loop* — blind, without
taking a new observation — before stopping to look and re-plan. At PushT's
10 fps, 32 steps means acting blind for ~3.2 s per plan vs ~0.8 s with the
paper's 8 — 4× fewer chances to notice the block slipped and correct course.
(Tradeoff: shorter chunks cost more inference per second and can jitter;
longer chunks are smoother but drift.)

**Run 1b (proposed): same command plus recipe-matching overrides** (verified
2026-07-28 via a 2-step dry run — resolved config matches the reference
values exactly; note the help text renders crop_shape as `[int int]` but the
space-separated form is rejected, use the bracket string):

```
  --policy.horizon=16 --policy.n_action_steps=8 \
  --policy.crop_shape="[84,84]" --policy.use_group_norm=true \
  --policy.pretrained_backbone_weights=null \
  --policy.use_separate_rgb_encoder_per_camera=false \
```

**Run 1b result (2026-07-28, run `diffusion_pusht_papercfg_20260728_1752`):**
**69.2% success / 0.970 avg max reward on 500 episodes — beats the reference
(65.4% / 0.955)**; gap fully explained by the config drift above. Trained in
3h15m (17.1 steps/s — faster than run 1 because the 84×84 crop + single
encoder shrink the model), wandb `snowpine007/lerobot/runs/v5fgp65v`.
In-training 50-ep evals: 36 → 50 → 50 → 44 → 60 → 48 → 52 → 56%. Final eval
artifacts: `/outputs/talking_to_robots/diffusion_pusht_papercfg_20260728_1752/eval`.
Phase-1 success bar (≥ ~65% on 500 episodes) met.

### Eval + visualization (after training)

```bash
# 50-episode eval of the final checkpoint (writes eval videos to output_dir)
RUN=<run_name>   # the run to evaluate — set explicitly, don't rely on the shell
RUN_DIR=/outputs/talking_to_robots/$RUN
# lerobot-eval does NOT refuse an existing output_dir — it silently overwrites
# eval_info.json and mixes videos. If $RUN_DIR/eval already exists, evaluate
# into a fresh sibling (e.g. eval_50ep_YYYYMMDD/) instead; never overwrite.
[ ! -e $RUN_DIR/eval ] || { echo "eval/ exists — pick a fresh dir"; exit 1; }
uv run lerobot-eval \
  --policy.path=$RUN_DIR/train/checkpoints/last/pretrained_model \
  --env.type=pusht --eval.n_episodes=50 --eval.batch_size=10 \
  --output_dir=$RUN_DIR/eval \
  2>&1 | tee $RUN_DIR/eval.log

# dataset visualization (rerun viewer)
uv run lerobot-dataset-viz --repo-id lerobot/pusht --episode-index 0
```

Success bar: reference checkpoint reports 65.4% success / 0.955 max-overlap
over 500 episodes; anywhere near that on 50 episodes = phase-1 win.

### Run 2: SmolVLA fine-tune (DONE 2026-08-11)

Language-conditioned run: fine-tune `lerobot/smolvla_base` (450M) on
`lerobot/svla_so100_pickplace` (owner-approved default; 50 episodes /
19,631 frames / 30 fps, SO-100 arm, cameras `top` + `wrist`, one task:
"Pick up the cube and place it in the box.").

**Command** (validated via 2-step dry runs + 300-step probe; probe:
1.48 steps/s, 14.4 GB VRAM):

```bash
cd ~/projects/talking_to_robots
RUN=smolvla_pickplace_$(date +%Y%m%d_%H%M)
RUN_DIR=/outputs/talking_to_robots/$RUN
mkdir -p $RUN_DIR
uv run lerobot-train \
  --policy.path=lerobot/smolvla_base --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --batch_size=64 --steps=20000 --log_freq=100 --save_freq=2500 \
  --output_dir=$RUN_DIR/train --job_name=$RUN --wandb.enable=true \
  2>&1 | tee $RUN_DIR/train.log
```

Gotchas learned:
- `smolvla_base` expects generic camera slots `camera1/2/3`; the dataset
  ships `top`/`wrist` → `--rename_map` required.
  `validate_visual_features_consistency` accepts a subset in either
  direction, so 2-of-3 cameras is fine; with this run's
  `empty_cameras: 0` the missing `camera3` is simply dropped — the model
  sees 2 images, identically at train and inference (padding with empty
  masked images exists but only activates when `empty_cameras > 0`;
  `modeling_smolvla.py:446`).
- The rename map is baked into the saved preprocessor
  (`policy_preprocessor.json` step 0) — inference with dataset-native
  keys just works; no manual renaming.
- Config comes FROM the checkpoint (`--policy.path`), not library
  defaults — the run-1 drift trap doesn't apply. Resolved snapshot
  verified anyway: action-expert-only training, frozen vision encoder,
  chunk 50 / n_action_steps 50, lr 1e-4 cosine.
- Do not append `; echo EXIT=$?` on the same line as the pipeline — it
  masks the exit code even with pipefail. Echo it as a separate command.

**Result (run `smolvla_pickplace_20260811_0823`):** trained clean, 20k
steps in 3h02m (1.83 steps/s, 14.4 GB), wandb `snowpine007/lerobot/runs/e0b0antu`.
Loss 0.46 (step 100) → 0.096 (2.5k) → 0.058 (5.4k) → 0.030 (10.5k) →
0.019 (15.4k) → **0.017 (20k)** — ≈65 epochs over 50 demos, normal for
BC on small demo sets. (Note: the log's `step:NK` labels round to the
nearest K; values here are matched to exact steps via checkpoint
timestamps.) 8 checkpoints + `last`.

**Eval — open-loop only.** This is a real-robot dataset: no simulator
exists for it, so there is no success-rate rollout (closed-loop eval
returns in Phase 2 where we own the env as code). Instead:
predicted-vs-ground-truth action chunks on 200 frames × 5 episodes
(`eval_openloop/` in the run dir; self-contained script + metrics +
plots). **Caveat stated on every artifact: the fine-tune used all 50
episodes (no held-out split), so this measures reproduction fidelity on
training demonstrations, NOT generalization.**
- Overall MAE 0.69 joint-units (RMSE 1.48); per-joint MAE = 2.2–4.7% of
  each joint's action std; gripper transients tracked.
- **Error grows along the 50-step chunk: MAE 0.51 (h=1) → 0.66 (h=30) →
  1.11 (h=50)** — flat for ~1 s then accelerating. Open-loop drift
  measured in our own model; empirical support for re-planning before
  chunk exhaustion (the run-1b lesson, now quantified).

## Rules of the road

- All run artifacts → `/outputs/talking_to_robots/<run_name>/` — one
  self-contained folder per run, logs included (never repo-local, no sibling
  files outside the run folder; layout in the Environment section).
- Per-run `.log` file via `tee` for every long job (see commands above).
- No GPU launches while owner sleeps; queue for morning go-ahead.
- Don't saturate the machine while the owner is actively using it (it's their
  daily-driver desktop) — modest footprint by default, full-throttle only on
  dedicated unattended runs with go-ahead.
- Feature branches + PRs after the initial scaffold commit; owner merges.
- Remote: `github.com/Davidxswang/talking_to_robots` (public, since 2026-07-27).

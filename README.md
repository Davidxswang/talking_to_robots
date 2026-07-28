# talking_to_robots

Robot learning with [LeRobot](https://github.com/huggingface/lerobot) — a fun,
community-facing project (not a competition).

- Phase 1: train/eval/visualize a policy on an existing sim benchmark (PushT).
- Phase 2: build our own MuJoCo task and wire it into LeRobot.
- Phase 3: upstream PRs to LeRobot.

## Layout

- `docs/plan.md` — the 3-phase arc + the phase-1 runbook (exact commands).
- `scripts/smoke_test.py` — CPU-only dataset + policy forward-pass check.
- `outputs/` — symlink to central `/outputs/talking_to_robots` (all run artifacts).

## Quick start

```bash
uv sync                                  # env (Python >=3.12, lerobot 0.6.0)
uv run python scripts/smoke_test.py      # CPU smoke test
```

Training/eval commands: see `docs/plan.md`.

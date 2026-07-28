"""CPU-only smoke test: dataset loads + Diffusion Policy forward pass works.

Run: uv run python scripts/smoke_test.py
No GPU is touched; this only proves imports, config, and shapes are sane
so the real GPU training launch does not die on a config error.
"""

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy

REPO_ID = "lerobot/pusht"
BATCH_SIZE = 2


def main() -> None:
    ds_meta = LeRobotDatasetMetadata(REPO_ID)
    print(f"dataset: {REPO_ID} episodes={ds_meta.total_episodes} fps={ds_meta.fps}")

    policy_cfg = PreTrainedConfig.get_choice_class("diffusion")(device="cpu")

    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    dataset = LeRobotDataset(REPO_ID, delta_timestamps=delta_timestamps)

    policy = make_policy(policy_cfg, ds_meta=ds_meta)
    policy.train()
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"policy: diffusion params={n_params / 1e6:.1f}M device=cpu")

    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    batch = next(iter(loader))
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"batch[{key}]: shape={tuple(value.shape)} dtype={value.dtype}")

    loss, _ = policy.forward(batch)
    print(f"forward OK, loss={loss.item():.4f}")


if __name__ == "__main__":
    main()

"""Custom dataset for iCardio echocardiogram videos.

Reads frames via byte-range seeks from WebDataset .tar shards using the
shard_index.pkl built by evaluation/build_index.py.  Returns:
    pixels: (T, C, H, W)  float32, ImageNet-normalized
    action: (T, 1)        float32 zeros (dummy — no actions for echo)
"""

import io
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class EchoDataset(Dataset):
    def __init__(self, uuids, shard_index, num_frames=9, img_size=224, train=True):
        self.uuids = list(uuids)
        self.shard_index = shard_index
        self.num_frames = num_frames
        self.img_size = img_size
        self.train = train

    def __len__(self):
        return len(self.uuids)

    def __getitem__(self, idx):
        uuid = self.uuids[idx]
        shard_path, offset, size, fmt = self.shard_index[uuid]

        try:
            with open(shard_path, "rb") as f:
                f.seek(offset)
                raw = f.read(size)
            buf = io.BytesIO(raw)
            frames = np.load(buf)["frames"] if fmt == "npz" else np.load(buf)
        except Exception:
            frames = np.zeros((self.num_frames, 336, 336, 3), dtype=np.uint8)

        T = len(frames)
        N = self.num_frames
        if T == 0:
            frames = np.zeros((N, 336, 336, 3), dtype=np.uint8)
        elif T < N:
            frames = np.concatenate([frames] + [frames[-1:]] * (N - T), axis=0)
        else:
            if self.train:
                start = np.random.randint(0, T - N + 1)
            else:
                start = (T - N) // 2
            frames = frames[start:start + N]

        # (T, H, W, C) uint8 → float32, ImageNet-normalize
        x = frames.astype(np.float32) / 255.0
        x = (x - _MEAN) / _STD           # (T, H, W, C)
        x = np.transpose(x, (0, 3, 1, 2))  # (T, C, H, W)

        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            xt = torch.from_numpy(x)
            xt = torch.nn.functional.interpolate(
                xt, size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False,
            )
            x = xt.numpy()

        pixels = torch.from_numpy(x).float()
        action = torch.zeros(N, 1, dtype=torch.float32)
        return {"pixels": pixels, "action": action}


def load_echo_dataset(
    shard_index_path,
    allowed_uuids_path,
    holdout_uuids_path=None,
    num_frames=9,
    img_size=224,
    train=True,
):
    """Build EchoDataset from shard index + UUID allowlist."""
    with open(shard_index_path, "rb") as f:
        shard_index = pickle.load(f)

    with open(allowed_uuids_path) as f:
        allowed = {line.strip() for line in f if line.strip()}

    holdout: set = set()
    if holdout_uuids_path and Path(holdout_uuids_path).exists():
        with open(holdout_uuids_path) as f:
            holdout = {line.strip() for line in f if line.strip()}

    candidates = [u for u in allowed if u not in holdout]
    uuids = [u for u in candidates if u in shard_index]
    n_holdout_removed = len(allowed) - len(candidates)
    n_not_indexed = len(candidates) - len(uuids)
    print(
        f"EchoDataset: {len(uuids):,} DICOMs "
        f"({len(allowed):,} allowed, {n_holdout_removed:,} removed by holdout, "
        f"{n_not_indexed:,} not in shard index)"
    )
    return EchoDataset(uuids, shard_index, num_frames=num_frames, img_size=img_size, train=train)

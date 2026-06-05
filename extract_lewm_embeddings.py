#!/usr/bin/env python3
"""Extract LeWorldModel embeddings for eval_icardio.py --probe-only.

Loads the LeWM ViT-Tiny encoder, processes each DICOM (9 frames at 224px),
averages CLS tokens over frames, and saves embeddings.pt in the format
expected by eval_icardio.py's probe pipeline.

Usage:
  conda activate echojepav2
  cd /home/mashrafimonon/iCardio
  python LeWorldModel/extract_lewm_embeddings.py \\
    --checkpoint checkpoints/lewm/echo_vitT_5pct/latest.pt \\
    --run-name lewm_5pct_e9 \\
    --device cuda:1

Then run probes:
  PYTHONPATH=EchoJEPAv2/EchoJEPA python EchoJEPAv2/evaluation/eval_icardio.py \\
    --run-name lewm_5pct_e9 --probe-only --device cuda:1
"""

import argparse
import gc
import io
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ViTConfig, ViTModel

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_SHARD_INDEX = Path("/home/mashrafimonon/iCardio/EchoJEPAv2/evaluation/shard_index.pkl")
_LABELS_DIR  = Path("/home/mashrafimonon/iCardio/output_with_labels/output")
_MANIFEST    = _LABELS_DIR / "manifest_clinical_findings_with_eval_labels.parquet"
_OUT_BASE    = Path("/home/mashrafimonon/iCardio/EchoJEPAv2/evaluation/results/icardio")


# ── Dataset ────────────────────────────────────────────────────────────────────

class DicomDataset(Dataset):
    def __init__(self, uuids, shard_index, num_frames=9, img_size=224):
        self.uuids = uuids
        self.shard_index = shard_index
        self.N = num_frames
        self.img_size = img_size

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
            frames = np.zeros((self.N, 336, 336, 3), dtype=np.uint8)

        T = len(frames)
        N = self.N
        if T == 0:
            frames = np.zeros((N, 336, 336, 3), dtype=np.uint8)
        elif T < N:
            frames = np.concatenate([frames] + [frames[-1:]] * (N - T), axis=0)
        else:
            start = (T - N) // 2
            frames = frames[start:start + N]

        x = frames.astype(np.float32) / 255.0
        x = (x - _MEAN) / _STD           # (N, H, W, C)
        x = np.transpose(x, (0, 3, 1, 2))  # (N, C, H, W)

        t = torch.from_numpy(x)
        if t.shape[-1] != self.img_size or t.shape[-2] != self.img_size:
            t = F.interpolate(t, size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False)
        return t.float(), uuid  # (N, C, H, W)


# ── Encoder ────────────────────────────────────────────────────────────────────

def load_encoder(checkpoint_path, device):
    print(f"Loading LeWM encoder from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    epoch = ckpt.get("epoch", "?")
    args  = ckpt.get("args", {})

    embed_dim  = args.get("embed_dim",  192)
    img_size   = args.get("img_size",   224)
    patch_size = args.get("patch_size", 14)

    cfg = ViTConfig(
        hidden_size=embed_dim,
        num_hidden_layers=12,
        num_attention_heads=3,
        intermediate_size=4 * embed_dim,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        image_size=img_size,
        patch_size=patch_size,
        num_channels=3,
        qkv_bias=True,
        add_pooling_layer=False,
    )
    encoder = ViTModel(cfg)

    enc_state = {k[len("encoder."):]: v
                 for k, v in ckpt["model"].items()
                 if k.startswith("encoder.")}
    msg = encoder.load_state_dict(enc_state, strict=True)
    print(f"  epoch={epoch}  embed_dim={embed_dim}  "
          f"missing={len(msg.missing_keys)}  unexpected={len(msg.unexpected_keys)}")

    del ckpt, enc_state; gc.collect()
    encoder.eval().to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder, img_size


# ── Extraction ─────────────────────────────────────────────────────────────────

def extract(checkpoint_path, run_name, device, batch_size, num_workers):
    out_dir = _OUT_BASE / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "embeddings.pt"

    if cache_path.exists():
        existing = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"Found {len(existing):,} cached embeddings in {cache_path}")
    else:
        existing = {}

    print("Loading shard index …")
    with open(_SHARD_INDEX, "rb") as f:
        shard_index = pickle.load(f)

    # Collect only DICOMs needed by eval tasks — same logic as eval_icardio.py
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "EchoJEPAv2" / "evaluation"))
    from eval_icardio import TASKS, build_study_dicom_map

    all_study_ids: set = set()
    for task_cfg in TASKS.values():
        df = pd.read_csv(_LABELS_DIR / task_cfg["csv"])
        all_study_ids.update(df["study_id"].dropna().tolist())
    print(f"  {len(all_study_ids):,} unique studies across {len(TASKS)} tasks")

    study_dicom_map = build_study_dicom_map(_MANIFEST, all_study_ids, shard_index, max_per_study=3)
    needed_uuids = [u for uuids in study_dicom_map.values() for u in uuids]
    uuids = [u for u in needed_uuids if u not in existing]
    print(f"  {len(uuids):,} DICOMs to extract  ({len(existing):,} already cached)")

    if not uuids:
        print("Nothing to do.")
        return

    # Sort by shard + offset for sequential HDD reads
    uuids.sort(key=lambda u: (shard_index[u][0], shard_index[u][1]))

    encoder, img_size = load_encoder(checkpoint_path, device)

    ds     = DicomDataset(uuids, shard_index, num_frames=9, img_size=img_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory="cuda" in device,
                        persistent_workers=num_workers > 0,
                        prefetch_factor=2 if num_workers > 0 else None)

    cache = dict(existing)
    autocast_dev = "cuda" if "cuda" in device else "cpu"
    save_every = 200

    for i, (clips, batch_uuids) in enumerate(tqdm(loader, desc="extracting", unit="batch")):
        # clips: (B, T, C, H, W)
        B, T, C, H, W = clips.shape
        frames = clips.view(B * T, C, H, W).to(device)

        with torch.no_grad():
            with torch.amp.autocast(autocast_dev, dtype=torch.bfloat16):
                out = encoder(pixel_values=frames)
                # CLS token: (B*T, 192)
                cls = out.last_hidden_state[:, 0, :]

        cls = cls.float().view(B, T, -1).mean(dim=1).cpu().numpy()  # (B, 192)
        for uuid, emb in zip(batch_uuids, cls):
            cache[uuid] = emb

        if (i + 1) % save_every == 0:
            tmp = cache_path.with_suffix(".tmp")
            torch.save(cache, tmp)
            tmp.rename(cache_path)
            print(f"  saved {len(cache):,} embeddings")

    torch.save(cache, cache_path)
    print(f"Done. Saved {len(cache):,} embeddings → {cache_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",   required=True)
    ap.add_argument("--run-name",     required=True,
                    help="Subdir under evaluation/results/icardio/")
    ap.add_argument("--device",       default="cuda:1")
    ap.add_argument("--batch-size",   type=int, default=32)
    ap.add_argument("--num-workers",  type=int, default=6)
    args = ap.parse_args()

    extract(args.checkpoint, args.run_name, args.device,
            args.batch_size, args.num_workers)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""LeWorldModel training on iCardio echocardiogram videos.

Dummy zero actions are passed (no real actions available).
Encoder: ViT-Tiny (192-dim, patch=14, img=224) via HuggingFace.
Loss: MSE prediction + SIGReg (weight=0.09).

Example (single GPU):
  conda activate echojepav2
  cd /home/mashrafimonon/iCardio/LeWorldModel
  python train_echo.py \\
    --shard-index ../EchoJEPAv2/evaluation/shard_index.pkl \\
    --train-uuids  ../EchoJEPAv2/training/train_dicoms_5pct.txt \\
    --holdout-uuids ../EchoJEPAv2/training/holdout_dicoms.txt \\
    --output-dir  ../checkpoints/lewm/echo_vit_tiny_5pct \\
    --devices cuda:0 \\
    --wandb-name lewm-echo-vitT-5pct

Example (multi-GPU via torchrun):
  torchrun --nproc_per_node=8 train_echo.py \\
    --shard-index ... --train-uuids ... --output-dir ... --batch-size 64
"""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from transformers import ViTConfig, ViTModel

sys.path.insert(0, str(Path(__file__).parent))
from echo_dataset import load_echo_dataset
from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


# ── Model construction ─────────────────────────────────────────────────────────

def build_vit_tiny(img_size=224, patch_size=14, embed_dim=192):
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
    return ViTModel(cfg)


def build_model(embed_dim=192, img_size=224, patch_size=14,
                action_dim=1, history_size=8):
    encoder = build_vit_tiny(img_size=img_size, patch_size=patch_size,
                              embed_dim=embed_dim)
    predictor = ARPredictor(
        num_frames=history_size,
        input_dim=embed_dim,
        hidden_dim=embed_dim,
        output_dim=embed_dim,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.0,
    )
    action_encoder = Embedder(input_dim=action_dim, emb_dim=embed_dim)
    projector = MLP(input_dim=embed_dim, hidden_dim=2048,
                    output_dim=embed_dim, norm_fn=nn.BatchNorm1d)
    pred_proj  = MLP(input_dim=embed_dim, hidden_dim=2048,
                     output_dim=embed_dim, norm_fn=nn.BatchNorm1d)
    return JEPA(encoder=encoder, predictor=predictor,
                action_encoder=action_encoder,
                projector=projector, pred_proj=pred_proj)


# ── Forward wrapper ────────────────────────────────────────────────────────────

class JEPAWrapper(nn.Module):
    def __init__(self, jepa, history_size, num_preds):
        super().__init__()
        self.jepa = jepa
        self.hs   = history_size
        self.np   = num_preds

    def forward(self, pixels, action):
        info = {"pixels": pixels, "action": action}
        output   = self.jepa.encode(info)
        emb      = output["emb"]      # (B, T, D)
        act_emb  = output["act_emb"]  # (B, T, D)
        pred_emb = self.jepa.predict(emb[:, :self.hs], act_emb[:, :self.hs])
        tgt_emb  = emb[:, self.np:]
        return pred_emb, tgt_emb, emb


# ── Training utilities ─────────────────────────────────────────────────────────

def forward_step(wrapper, sigreg, batch, device, sigreg_w):
    pixels = batch["pixels"].to(device)
    action = torch.nan_to_num(batch["action"].to(device), 0.0)
    pred_emb, tgt_emb, emb = wrapper(pixels, action)
    pred_loss   = (pred_emb - tgt_emb).pow(2).mean()
    sigreg_loss = sigreg(emb.transpose(0, 1))
    loss        = pred_loss + sigreg_w * sigreg_loss
    return loss, pred_loss.detach(), sigreg_loss.detach()


def lr_schedule(epoch, max_epochs, base_lr, min_lr=1e-7, warmup=5):
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    t = (epoch - warmup) / max(1, max_epochs - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-index",   required=True)
    ap.add_argument("--train-uuids",   required=True)
    ap.add_argument("--holdout-uuids", default=None)
    ap.add_argument("--output-dir",    required=True)

    ap.add_argument("--epochs",        type=int,   default=50)
    ap.add_argument("--batch-size",    type=int,   default=64,
                    help="Per-GPU batch size")
    ap.add_argument("--lr",            type=float, default=5e-5)
    ap.add_argument("--weight-decay",  type=float, default=1e-3)
    ap.add_argument("--warmup-epochs", type=int,   default=5)
    ap.add_argument("--grad-clip",     type=float, default=1.0)

    ap.add_argument("--devices", nargs="+", default=["cuda:0"],
                    help="Used for single-GPU only; ignored when torchrun sets LOCAL_RANK")
    ap.add_argument("--num-workers",   type=int,   default=6)
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--prefetch",      type=int,   default=3)

    ap.add_argument("--history-size",  type=int,   default=8)
    ap.add_argument("--num-preds",     type=int,   default=1)
    ap.add_argument("--img-size",      type=int,   default=224)
    ap.add_argument("--embed-dim",     type=int,   default=192)
    ap.add_argument("--patch-size",    type=int,   default=14)

    ap.add_argument("--sigreg-weight",   type=float, default=0.09)
    ap.add_argument("--sigreg-knots",    type=int,   default=17)
    ap.add_argument("--sigreg-num-proj", type=int,   default=1024)

    ap.add_argument("--train-split",   type=float, default=0.95)

    ap.add_argument("--wandb-project", default="lewm-echo")
    ap.add_argument("--wandb-name",    default="lewm-echo-5pct")
    ap.add_argument("--wandb-entity",  default="anaatef9-mbzuai")
    ap.add_argument("--no-wandb",      action="store_true")
    ap.add_argument("--resume",        default=None)
    args = ap.parse_args()

    # ── DDP / device setup ─────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp     = local_rank >= 0 and world_size > 1

    if is_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        # Single-GPU path: honour --devices[0]
        device = torch.device(args.devices[0])
        local_rank = 0

    is_master = local_rank == 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.output_dir)
    if is_master:
        out.mkdir(parents=True, exist_ok=True)

    num_frames = args.history_size + args.num_preds

    # ── Dataset ────────────────────────────────────────────────────────────────
    if is_master:
        print("Loading dataset …")
    full_ds = load_echo_dataset(
        shard_index_path=args.shard_index,
        allowed_uuids_path=args.train_uuids,
        holdout_uuids_path=args.holdout_uuids,
        num_frames=num_frames,
        img_size=args.img_size,
        train=True,
    )
    rng = torch.Generator().manual_seed(args.seed)
    n_val   = max(1, int(len(full_ds) * (1 - args.train_split)))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=rng)
    val_ds.dataset.train = False

    if is_master:
        print(f"  {len(full_ds):,} DICOMs  train {n_train:,}  val {n_val:,}")

    nw = args.num_workers
    pf = args.prefetch if nw > 0 else None

    train_sampler = DistributedSampler(train_ds, shuffle=True,  seed=args.seed) if is_ddp else None
    val_sampler   = DistributedSampler(val_ds,   shuffle=False, seed=args.seed) if is_ddp else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=nw, drop_last=True, pin_memory=False,
        persistent_workers=nw > 0, prefetch_factor=pf,
        generator=(rng if train_sampler is None else None),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        sampler=val_sampler, shuffle=False,
        num_workers=nw, drop_last=False, pin_memory=False,
        persistent_workers=nw > 0, prefetch_factor=pf,
    )

    if is_master:
        print(f"  {len(train_loader)} train batches / {len(val_loader)} val batches per rank")

    # ── Model ──────────────────────────────────────────────────────────────────
    if is_master:
        print("Building model …")
    model  = build_model(embed_dim=args.embed_dim, img_size=args.img_size,
                         patch_size=args.patch_size, action_dim=1,
                         history_size=args.history_size).to(device)
    sigreg = SIGReg(knots=args.sigreg_knots,
                    num_proj=args.sigreg_num_proj).to(device)

    if is_master:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  {total_params:,} trainable parameters")

    wrapper = JEPAWrapper(model, args.history_size, args.num_preds)
    if is_ddp:
        # SyncBatchNorm keeps BN stats consistent across ranks
        wrapper = nn.SyncBatchNorm.convert_sync_batchnorm(wrapper)
        wrapper = DDP(wrapper, device_ids=[local_rank])
        if is_master:
            print(f"  DDP on {world_size} GPUs")

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(sigreg.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        if is_master:
            print(f"  Resumed from epoch {start_epoch}")

    # ── WandB (master only) ────────────────────────────────────────────────────
    use_wb = _WANDB and not args.no_wandb and is_master
    if use_wb:
        wandb.init(project=args.wandb_project, name=args.wandb_name,
                   entity=args.wandb_entity, config=vars(args))

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        lr = lr_schedule(epoch, args.epochs, args.lr,
                         warmup=args.warmup_epochs)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # train
        wrapper.train()
        t_loss, t_pred, t_sreg = [], [], []
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss, pl, sl = forward_step(wrapper, sigreg, batch, device, args.sigreg_weight)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            t_loss.append(loss.item())
            t_pred.append(pl.item())
            t_sreg.append(sl.item())
            if is_master and step % 100 == 0:
                print(f"  E{epoch+1:03d}/{args.epochs} "
                      f"[{step:4d}/{len(train_loader)}] "
                      f"loss={loss.item():.4f} "
                      f"pred={pl.item():.4f} "
                      f"sigreg={sl.item():.4f} "
                      f"lr={lr:.2e}", flush=True)

        # val
        wrapper.eval()
        v_loss, v_pred, v_sreg = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                loss, pl, sl = forward_step(wrapper, sigreg, batch, device, args.sigreg_weight)
                v_loss.append(loss.item())
                v_pred.append(pl.item())
                v_sreg.append(sl.item())

        if is_master:
            print(f"E{epoch+1:03d} | "
                  f"train loss={np.mean(t_loss):.4f} pred={np.mean(t_pred):.4f} "
                  f"sreg={np.mean(t_sreg):.4f} | "
                  f"val  loss={np.mean(v_loss):.4f} pred={np.mean(v_pred):.4f} "
                  f"sreg={np.mean(v_sreg):.4f}")

            if use_wb:
                wandb.log({
                    "train/loss": np.mean(t_loss), "train/pred_loss": np.mean(t_pred),
                    "train/sigreg_loss": np.mean(t_sreg),
                    "val/loss":   np.mean(v_loss), "val/pred_loss":   np.mean(v_pred),
                    "val/sigreg_loss":   np.mean(v_sreg),
                    "lr": lr, "epoch": epoch + 1,
                })

            # checkpoint (keep last 2 + latest)
            ckpt_state = {"epoch": epoch + 1, "model": model.state_dict(),
                          "optimizer": optimizer.state_dict(), "args": vars(args)}
            torch.save(ckpt_state, out / f"epoch_{epoch+1:03d}.pt")
            torch.save(ckpt_state, out / "latest.pt")
            old = sorted(out.glob("epoch_*.pt"))[:-2]
            for p in old:
                p.unlink()

    if use_wb:
        wandb.finish()
    if is_master:
        print("Done.")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

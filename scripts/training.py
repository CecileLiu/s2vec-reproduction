#!/usr/bin/env python3
"""
training.py — Stage 3: MAE pre-training on rasterized S2 images.

Input:  ../data/processed/images.npy   [N, 16, 16, F]  (from rasterize.py)
Output: ../checkpoints/mae_best.pt     best checkpoint (lowest val loss)
        ../data/processed/norm_stats.npz  per-feature mean/std for inference

Run from the scripts/ directory:
    python training.py
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path("../data/processed")
CKPT_DIR = Path("../checkpoints")

# ── Hyperparameters ─────────────────────────────────────────────────────────────

GRID_SIZE = 16
MASK_RATIO = 0.75

ENCODER_DIM = 384
ENCODER_DEPTH = 6
ENCODER_HEADS = 6
DECODER_DIM = 192
DECODER_DEPTH = 4
DECODER_HEADS = 4
MLP_RATIO = 4.0

EPOCHS = 200
BATCH_SIZE = 64
BASE_LR = 1.5e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 20
SAVE_EVERY = 50          # also checkpoint every N epochs

VAL_FRAC = 0.1
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Dataset ────────────────────────────────────────────────────────────────────

class S2ImageDataset(Dataset):
    """Normalized [N_patches, F] patch sequences for MAE training."""

    def __init__(self, images: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
        N, H, W, F = images.shape
        normed = (images - mean) / (std + 1e-6)
        # Flatten spatial dims → token sequence
        self.data = torch.from_numpy(
            normed.reshape(N, H * W, F).astype(np.float32)
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]   # [N_patches, F]


# ── Position embeddings ─────────────────────────────────────────────────────────

def sincos_2d_pos_embed(dim: int, h: int, w: int) -> torch.Tensor:
    """Fixed 2D sin-cos position embeddings.  Returns [h*w, dim]."""
    assert dim % 4 == 0
    d = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(d, dtype=torch.float32) / d))

    rows = torch.arange(h, dtype=torch.float32).unsqueeze(1) * omega   # [h, d]
    cols = torch.arange(w, dtype=torch.float32).unsqueeze(1) * omega   # [w, d]

    row_enc = torch.cat([rows.sin(), rows.cos()], dim=-1)   # [h, 2d]
    col_enc = torch.cat([cols.sin(), cols.cos()], dim=-1)   # [w, 2d]

    embed = torch.zeros(h, w, dim)
    embed[:, :, :2*d] = row_enc.unsqueeze(1).expand(h, w, 2*d)
    embed[:, :, 2*d:] = col_enc.unsqueeze(0).expand(h, w, 2*d)
    return embed.reshape(h * w, dim)                        # [h*w, dim]


# ── Transformer block ───────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm1(x)
        x = x + self.attn(n, n, n, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ── MAE model ──────────────────────────────────────────────────────────────────

class MAE(nn.Module):
    """Masked Autoencoder for S2 geospatial patch embeddings.

    Input:  [B, N_patches, F]  — normalized feature count vectors
    Output: reconstruction of masked patches + binary mask
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_patches = GRID_SIZE * GRID_SIZE
        self.n_keep = int(self.num_patches * (1 - MASK_RATIO))

        # ── Encoder ────────────────────────────────────────────────────────────
        self.patch_embed = nn.Linear(feature_dim, ENCODER_DIM)
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(ENCODER_DIM, ENCODER_HEADS, MLP_RATIO)
            for _ in range(ENCODER_DEPTH)
        ])
        self.encoder_norm = nn.LayerNorm(ENCODER_DIM)

        # ── Decoder ────────────────────────────────────────────────────────────
        self.enc_to_dec = nn.Linear(ENCODER_DIM, DECODER_DIM, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, DECODER_DIM))
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(DECODER_DIM, DECODER_HEADS, MLP_RATIO)
            for _ in range(DECODER_DEPTH)
        ])
        self.decoder_norm = nn.LayerNorm(DECODER_DIM)
        self.decoder_pred = nn.Linear(DECODER_DIM, feature_dim, bias=True)

        # Fixed position embeddings (registered as buffers — saved with state_dict)
        self.register_buffer(
            "enc_pos", sincos_2d_pos_embed(ENCODER_DIM, GRID_SIZE, GRID_SIZE)
        )
        self.register_buffer(
            "dec_pos", sincos_2d_pos_embed(DECODER_DIM, GRID_SIZE, GRID_SIZE)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _mask(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly select n_keep patches; return (x_vis, mask, ids_restore)."""
        B, N, D = x.shape
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = noise.argsort(dim=1)                  # low noise → visible
        ids_restore = ids_shuffle.argsort(dim=1)            # inverse permutation

        ids_keep = ids_shuffle[:, :self.n_keep]
        x_vis = x.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, N, device=x.device)            # 1 = masked
        mask.scatter_(1, ids_keep, 0.0)

        return x_vis, mask, ids_restore

    def encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed + mask + encode visible patches."""
        x = self.patch_embed(x) + self.enc_pos             # [B, N, enc_dim]
        x_vis, mask, ids_restore = self._mask(x)

        for blk in self.encoder_blocks:
            x_vis = blk(x_vis)
        x_vis = self.encoder_norm(x_vis)
        return x_vis, mask, ids_restore

    def encode_no_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Encode all patches without masking — used for inference / embedding extraction.

        Returns [B, N_patches, ENCODER_DIM].  Each token corresponds to the
        cell at the matching position in the 16×16 raster grid.
        """
        x = self.patch_embed(x) + self.enc_pos             # [B, N, enc_dim]
        for blk in self.encoder_blocks:
            x = blk(x)
        return self.encoder_norm(x)

    def decode(self, latent: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """Project encoder output, fill mask tokens, decode all patches."""
        B = latent.shape[0]
        latent = self.enc_to_dec(latent)                    # [B, n_vis, dec_dim]

        n_masked = self.num_patches - latent.shape[1]
        mask_tokens = self.mask_token.expand(B, n_masked, -1)

        # Restore original token order before adding position embeddings
        x = torch.cat([latent, mask_tokens], dim=1)
        x = x.gather(1, ids_restore.unsqueeze(-1).expand(-1, -1, DECODER_DIM))
        x = x + self.dec_pos                               # [B, N, dec_dim]

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        return self.decoder_pred(x)                        # [B, N, F]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent, mask, ids_restore = self.encode(x)
        pred = self.decode(latent, ids_restore)
        return pred, mask                                   # both [B, N, *]


# ── Loss ───────────────────────────────────────────────────────────────────────

def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE averaged over masked positions only."""
    loss = F.mse_loss(pred, target, reduction="none").mean(dim=-1)   # [B, N]
    return (loss * mask).sum() / mask.sum()


# ── LR schedule ────────────────────────────────────────────────────────────────

def set_lr(optimizer: torch.optim.Optimizer, epoch: int) -> float:
    if epoch < WARMUP_EPOCHS:
        lr = BASE_LR * (epoch + 1) / WARMUP_EPOCHS
    else:
        t = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
        lr = BASE_LR * 0.5 * (1.0 + math.cos(math.pi * t))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ── Train / val one epoch ───────────────────────────────────────────────────────

def run_epoch(
    model: MAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    train = optimizer is not None
    model.train(train)
    total = 0.0
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = batch.to(DEVICE)
            pred, mask = model(batch)
            loss = masked_mse(pred, batch, mask)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total += loss.item() * len(batch)
    return total / len(loader.dataset)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # 1. Load rasterized images
    images_path = PROCESSED_DIR / "images.npy"
    if not images_path.exists():
        raise FileNotFoundError(f"{images_path} not found — run rasterize.py first")

    images = np.load(images_path)                   # [N, 16, 16, F]
    N, H, W, F = images.shape
    log.info("Loaded images: %d × %d×%d × %d features  (device=%s)", N, H, W, F, DEVICE.upper())

    # 2. Train / val split
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(N)
    n_val = max(1, int(N * VAL_FRAC))
    train_imgs, val_imgs = images[idx[n_val:]], images[idx[:n_val]]
    log.info("  Train: %d  Val: %d", len(train_imgs), len(val_imgs))

    # 3. Per-feature z-score stats from training images only
    flat = train_imgs.reshape(-1, F).astype(np.float64)
    mean = flat.mean(axis=0).astype(np.float32)
    std  = flat.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0           # constant features → don't divide by zero

    np.savez(PROCESSED_DIR / "norm_stats.npz", mean=mean, std=std)
    log.info("Saved normalization stats → %s", PROCESSED_DIR / "norm_stats.npz")

    # 4. DataLoaders (num_workers=0 for cross-platform safety)
    train_ds = S2ImageDataset(train_imgs, mean, std)
    val_ds   = S2ImageDataset(val_imgs,   mean, std)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 5. Model + optimizer
    model = MAE(feature_dim=F).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("MAE  enc=%d×d%d  dec=%d×d%d  params=%s",
             ENCODER_DEPTH, ENCODER_DIM, DECODER_DEPTH, DECODER_DIM, f"{n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95)
    )

    # 6. Training loop
    best_val = float("inf")
    log.info("Training for %d epochs ...", EPOCHS)

    for epoch in range(EPOCHS):
        lr = set_lr(optimizer, epoch)
        train_loss = run_epoch(model, train_loader, optimizer)
        val_loss   = run_epoch(model, val_loader,   None)

        log.info(
            "epoch %3d/%d  train=%.4f  val=%.4f  lr=%.2e",
            epoch + 1, EPOCHS, train_loss, val_loss, lr,
        )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "feature_dim": F,
            "config": dict(
                encoder_dim=ENCODER_DIM, encoder_depth=ENCODER_DEPTH, encoder_heads=ENCODER_HEADS,
                decoder_dim=DECODER_DIM, decoder_depth=DECODER_DEPTH, decoder_heads=DECODER_HEADS,
                mask_ratio=MASK_RATIO, grid_size=GRID_SIZE,
            ),
        }

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, CKPT_DIR / "mae_best.pt")

        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save(ckpt, CKPT_DIR / f"mae_epoch{epoch+1:04d}.pt")

    log.info("Done.  Best val loss: %.4f  → %s", best_val, CKPT_DIR / "mae_best.pt")


if __name__ == "__main__":
    main()

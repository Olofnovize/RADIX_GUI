"""Deployment inference: DINOv2+DPT on 3-channel microscopy samples."""

import csv
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import tifffile
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.lightning_module import TimmSemanticModule
from src.model_classes import (
    unet_semantic_to_bio7,
    BIO_7_NAMES, BIO_7_COLORS_RGB,
)

# ═══════════════════════════════════════════════════════════════════════════════
# EDIT THESE PATHS
# ═══════════════════════════════════════════════════════════════════════════════

DATA_DIR = None
MODEL_DIR = None
OUTPUT_DIR = None
SAVE_VIS = True


# ═══════════════════════════════════════════════════════════════════════════════

IMG_SIZE = 1024
CHANNEL_NAMES = ["TRITC", "FITC", "DAPI"]

CHANNEL_ALIASES = {
    "FITC": ["FITC", "GFP"],
    "TRITC": ["TRITC", "RFP"],
    "DAPI": ["DAPI", "CW", "CALCOFLUOR"],
}

MEASURED_REGIONS = ["Epidermis", "Exodermis", "Endodermis", "Cortex", "Vascular"]
CROP_MARGIN_PCT = 0.05


def find_checkpoint(model_dir: Path) -> Path:
    if model_dir.is_file() and model_dir.suffix == ".ckpt":
        return model_dir

    if model_dir.is_dir():
        ckpt_dir = model_dir / "checkpoints" if (model_dir / "checkpoints").is_dir() else model_dir
        candidates = sorted(ckpt_dir.glob("best-*.ckpt"))
        if candidates:
            return candidates[-1]

        any_ckpts = sorted(ckpt_dir.glob("*.ckpt"))
        if any_ckpts:
            return any_ckpts[-1]

    raise FileNotFoundError(f"No checkpoint found under {model_dir}")


def discover_samples(data_dir: Path):
    samples = []
    tif_exts = ("*.tif", "*.tiff", "*.ome.tif", "*.ome.tiff")

    for sub in sorted(data_dir.iterdir()):
        if not sub.is_dir() or sub.name.startswith("."):
            continue

        all_tifs = []
        for ext in tif_exts:
            all_tifs.extend(sub.glob(ext))

        channels = {}

        for ch in CHANNEL_NAMES:
            aliases = CHANNEL_ALIASES.get(ch.upper(), [ch])
            found = None

            for alias in aliases:
                alias_low = alias.lower()
                tokens = (
                    f"_{alias_low}.",
                    f"_{alias_low}_",
                    f"({alias_low})",
                    f"-{alias_low}-",
                    f"-{alias_low}.",
                )

                for f in all_tifs:
                    fname = f.name.lower()
                    if any(tok in fname for tok in tokens):
                        found = f
                        break

                if found is not None:
                    break

            if found is not None:
                channels[ch] = found

        if len(channels) == 3:
            samples.append((sub.name, channels))
        else:
            missing = set(CHANNEL_NAMES) - set(channels)
            print(f"  [skip] {sub.name}: missing channels {missing}")

    return samples


def load_channel(path: Path) -> np.ndarray:
    img = tifffile.imread(str(path))
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.float32)


def load_sample_3ch(channels: dict):
    stacks = [load_channel(channels[ch]) for ch in CHANNEL_NAMES]
    return np.stack(stacks, axis=-1)


def normalize_percentile(img: np.ndarray, p_low=1.0, p_high=99.5):
    out = np.empty_like(img, dtype=np.float32)
    for c in range(img.shape[-1]):
        ch = img[..., c]
        lo = np.percentile(ch, p_low)
        hi = np.percentile(ch, p_high)
        if hi > lo:
            out[..., c] = np.clip((ch - lo) / (hi - lo), 0, 1)
        else:
            out[..., c] = 0.0
    return out


def to_uint8(img01: np.ndarray) -> np.ndarray:
    return (np.clip(img01, 0, 1) * 255).astype(np.uint8)


def run_inference(model, device, img_norm: np.ndarray) -> np.ndarray:
    h, w = img_norm.shape[:2]
    resized = cv2.resize(img_norm, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    return cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)


def find_root_bbox(sem_mask: np.ndarray, margin_pct: float = CROP_MARGIN_PCT):
    fg = sem_mask >= 1
    if not fg.any():
        return None

    ys, xs = np.where(fg)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    bh, bw = y1 - y0, x1 - x0
    my = int(round(bh * margin_pct))
    mx = int(round(bw * margin_pct))

    H, W = sem_mask.shape
    y0 = max(0, y0 - my)
    y1 = min(H, y1 + my)
    x0 = max(0, x0 - mx)
    x1 = min(W, x1 + mx)

    return y0, y1, x0, x1


def overlay_bio7(rgb_uint8, bio7_dict, alpha=0.5):
    result = rgb_uint8.astype(np.float32)

    for cls_id, name in enumerate(BIO_7_NAMES):
        m = bio7_dict.get(name)
        if m is None:
            continue

        mask = m > 0
        if not mask.any():
            continue

        color = np.array(BIO_7_COLORS_RGB[cls_id], dtype=np.float32)
        result[mask] = (1 - alpha) * result[mask] + alpha * color

    return np.clip(result, 0, 255).astype(np.uint8)


def save_vis(rgb_u8, bio7, sample_name, out_path):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    axes[0, 0].imshow(rgb_u8)
    axes[0, 0].set_title("Input")
    axes[0, 0].axis("off")

    for i, name in enumerate(BIO_7_NAMES):
        row = (i + 1) // 4
        col = (i + 1) % 4
        if row >= 2 or col >= 4:
            break

        ax = axes[row, col]
        mask = bio7[name] > 0
        frac = float(mask.sum()) / mask.size * 100

        color = np.array(BIO_7_COLORS_RGB[i], dtype=np.float32)
        overlay = rgb_u8.astype(np.float32).copy()
        overlay[mask] = 0.4 * overlay[mask] + 0.6 * color

        ax.imshow(overlay.clip(0, 255).astype(np.uint8))
        ax.set_title(f"{name} ({frac:.1f}%)")
        ax.axis("off")

    ax = axes[1, 3]
    ax.imshow(overlay_bio7(rgb_u8, bio7))
    ax.set_title("All classes")
    ax.axis("off")

    fig.suptitle(f"{sample_name} — DINOv2+DPT", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def measure(sample_name: str, raw_img: np.ndarray, bio7: dict) -> dict:
    h, w = raw_img.shape[:2]
    mh, mw = bio7["Whole Root"].shape[:2]

    if (mh, mw) != (h, w):
        bio7 = {k: cv2.resize(v, (w, h), interpolation=cv2.INTER_NEAREST) for k, v in bio7.items()}

    wr = bio7["Whole Root"]
    aer = bio7["Aerenchyma"]
    aer_ratio = float(aer.sum()) / max(int(wr.sum()), 1)

    row = {
        "sample_id": sample_name,
        "aerenchyma_ratio": aer_ratio,
    }

    ch_idx = {"TRITC": 0, "FITC": 1, "DAPI": 2}

    for region in MEASURED_REGIONS:
        mask = bio7[region] > 0

        # For cortex intensity, exclude aerenchyma spaces
        if region == "Cortex":
            mask = mask & ~(bio7["Aerenchyma"] > 0)

        for ch in ("DAPI", "FITC", "TRITC"):
            channel_img = raw_img[..., ch_idx[ch]]
            val = float(channel_img[mask].mean()) if mask.any() else 0.0
            row[f"{region.lower()}_{ch}"] = val

    return row


def parse_args():
    parser = argparse.ArgumentParser(description="Run RootLensX segmentation inference.")

    parser.add_argument(
        "--data-dir",
        required=True,
        help="Folder containing per-sample raw image folders with TRITC/FITC/DAPI TIFFs.",
    )

    parser.add_argument(
        "--model-dir",
        required=True,
        help="Checkpoint file or folder containing .ckpt files.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output folder for predictions, visualizations, and measurements.",
    )

    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Disable saving visualization PNGs.",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum number of samples to process.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.output_dir).resolve()

    save_visualizations = not args.no_vis

    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (out_dir / "vis").mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt = find_checkpoint(model_dir)
    print(f"Loading checkpoint: {ckpt}")

    model = TimmSemanticModule.load_from_checkpoint(str(ckpt), map_location=device)
    model.eval().to(device)

    samples = discover_samples(data_dir)
    print(f"Discovered {len(samples)} samples in {data_dir}")

    if not samples:
        print("No samples found.")
        return

    rows = []

    for sample_name, channels in samples:
        print(f"-- {sample_name}")

        raw = load_sample_3ch(channels)
        norm = normalize_percentile(raw)

        sem_coarse = run_inference(model, device, norm)
        bbox = find_root_bbox(sem_coarse, margin_pct=CROP_MARGIN_PCT)

        if bbox is None:
            print("  [warn] no root detected; using full image")
            raw_crop = raw
            norm_crop = norm
        else:
            y0, y1, x0, x1 = bbox
            raw_crop = raw[y0:y1, x0:x1]
            norm_crop = normalize_percentile(raw_crop)

            print(
                f"  root bbox: y=[{y0}:{y1}] x=[{x0}:{x1}]  "
                f"crop {raw_crop.shape[0]}x{raw_crop.shape[1]} "
                f"(full {raw.shape[0]}x{raw.shape[1]})"
            )

        sem_crop = run_inference(model, device, norm_crop)

        # Put cropped prediction back into full original image coordinates
        sem_full = np.zeros(raw.shape[:2], dtype=np.uint8)

        if bbox is None:
            sem_full = sem_crop
        else:
            y0, y1, x0, x1 = bbox
            sem_full[y0:y1, x0:x1] = sem_crop

        # Cropped mask for Yifei-style visualization and measurement
        bio7_crop = unet_semantic_to_bio7(sem_crop, *sem_crop.shape)

        # Full-size mask for downstream post-processing on raw images
        np.save(out_dir / "predictions" / f"{sample_name}.npy", sem_full)

        bio7_full = unet_semantic_to_bio7(sem_full, *sem_full.shape)

        if save_visualizations:
            save_vis(
                to_uint8(norm),
                bio7_full,
                sample_name,
                out_dir / "vis" / f"{sample_name}.png"
            )

        rows.append(measure(sample_name, raw_crop, bio7_crop))

    cols = ["sample_id", "aerenchyma_ratio"]

    for region in MEASURED_REGIONS:
        for ch in ("DAPI", "FITC", "TRITC"):
            cols.append(f"{region.lower()}_{ch}")

    csv_path = out_dir / "measurements.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nSaved {len(rows)} samples:")
    print(f"  predictions: {out_dir / 'predictions'}")
    print(f"  visualizations: {out_dir / 'vis'}")
    print(f"  measurements: {csv_path}")


if __name__ == "__main__":
    main()
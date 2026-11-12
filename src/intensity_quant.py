import numpy as np
import pandas as pd
from pathlib import Path
from skimage import filters
from skimage.measure import label, regionprops
from .io_utils import load_sample
from .annotation_utils import parse_yolo_annotations, polygon_to_mask
from .model_classes import get_filled_classes, yolo_overlap_false_to_bio7

# Tissues quantified by the barrier-intensity tools below. Cortex excludes
# Aerenchyma pixels (air space, not cortex tissue) via _tissue_mask().
BARRIER_TISSUES = ["Endodermis", "Exodermis", "Cortex", "Epidermis", "Vascular"]


def _tissue_mask(bio7, tissue):
    mask = bio7[tissue] > 0
    if tissue == "Cortex":
        mask = mask & ~(bio7.get("Aerenchyma", np.zeros_like(mask)) > 0)
    return mask


def normalize_channel(img, p_low=1.0, p_high=99.5):
    img = img.astype(np.float32)
    lo = np.percentile(img, p_low)
    hi = np.percentile(img, p_high)

    if hi > lo:
        return np.clip((img - lo) / (hi - lo), 0, 1)

    return np.zeros_like(img, dtype=np.float32)


def normalize_percentile_rgb(raw, p_low=0.5, p_high=99.5):
    out = np.zeros_like(raw, dtype=np.float32)

    for c in range(raw.shape[-1]):
        out[..., c] = normalize_channel(raw[..., c], p_low=p_low, p_high=p_high)

    return out


def _load_gt_bio7(sample_name, data_dir, gt_mask_dir, channels):
    """Load raw image + bio7 mask from GT YOLO .txt annotations."""
    import tifffile

    ch_order = list(channels)
    data_dir = Path(data_dir)
    gt_mask_dir = Path(gt_mask_dir)

    # Find raw TIF files
    raw_channels = []
    sample_dir = None
    for p in data_dir.rglob(f"{sample_name}_TRITC.tif"):
        sample_dir = p.parent
        break

    if sample_dir is None:
        raise FileNotFoundError(f"No image folder found for {sample_name}")

    for ch in ch_order:
        ch_path = sample_dir / f"{sample_name}_{ch}.tif"
        if not ch_path.exists():
            raise FileNotFoundError(f"Missing {ch} image: {ch_path}")
        img = tifffile.imread(str(ch_path))
        img = np.squeeze(img)
        if img.ndim == 3:
            img = img.max(axis=0)
        raw_channels.append(img.astype(np.float32))

    raw = np.stack(raw_channels, axis=-1)
    h, w = raw.shape[:2]

    # Find GT annotation
    matches = list(gt_mask_dir.rglob(f"*{sample_name}.txt"))
    if not matches:
        raise FileNotFoundError(f"No GT annotation found for {sample_name}")

    anns = parse_yolo_annotations(matches[0], w, h)
    if not anns:
        raise ValueError(f"Empty annotation file for {sample_name}")

    masks = np.stack(
        [polygon_to_mask(a["polygon"], h, w) for a in anns], axis=0
    ).astype(np.uint8)
    labels = np.array([a["class_id"] for a in anns], dtype=np.int32)

    filled = get_filled_classes(masks, labels, h, w)
    bio7 = yolo_overlap_false_to_bio7(filled, h, w)

    return raw, bio7


def compute_threshold_percentile_suggestions(
    samples,
    data_dir,
    pred_dir,
    channels,
    percentiles=(50, 60, 70, 80, 90, 95, 99),
    mask_source="Prediction (.npy)",
    gt_mask_dir=None,
):
    """
    Compute raw-intensity percentile suggestions across all loaded samples,
    separately for each barrier tissue and FITC/TRITC.

    mask_source : "Prediction (.npy)" or "GT YOLO (.txt)"
    gt_mask_dir : required when mask_source == "GT YOLO (.txt)"
    """

    ch_idx = {"TRITC": 0, "FITC": 1, "DAPI": 2}
    rows = []
    errors = []

    for tissue in BARRIER_TISSUES:
        for ch in ["FITC", "TRITC"]:
            vals_all = []

            for sample_name in samples:
                try:
                    if mask_source == "GT YOLO (.txt)":
                        if not gt_mask_dir:
                            raise ValueError("gt_mask_dir must be set for GT mode")
                        raw_i, bio7_i = _load_gt_bio7(
                            sample_name, data_dir, gt_mask_dir, channels
                        )
                    else:
                        raw_i, bio7_i = load_sample(
                            sample_name, data_dir, pred_dir, channels
                        )

                    mask_i = _tissue_mask(bio7_i, tissue)
                    if not mask_i.any():
                        continue

                    img_i = raw_i[..., ch_idx[ch]].astype(np.float32)
                    vals = img_i[mask_i]

                    if vals.size:
                        vals_all.append(vals)

                except Exception as e:
                    errors.append(f"{sample_name}: {e}")
                    continue

            if not vals_all:
                continue

            vals_all = np.concatenate(vals_all)

            row = {
                "Tissue": tissue,
                "Channel": ch,
                "N_pixels": int(vals_all.size),
                "N_samples": len(vals_all),
            }

            for p in percentiles:
                row[f"p{p}"] = float(np.percentile(vals_all, p))

            rows.append(row)

    result = pd.DataFrame(rows)
    result._errors = errors   # attach for caller inspection
    return result


def measure_root_morphometrics(bio7, um_per_px=None):
    """
    Compute root geometry and tissue composition from bio7 masks.

    Parameters
    ----------
    bio7 : dict  {tissue_name: (H,W) uint8 mask}
        As returned by yolo_overlap_false_to_bio7 etc.
    um_per_px : float or None
        Pixel size in µm. If provided, area values are also reported in µm².
        If None, only pixel-based values are returned.

    Returns
    -------
    dict  — flat key→value, all floats or int. NaN where mask is empty.
    """
    rows = {}

    # ── Root-level geometry ──────────────────────────────────────────────────
    wr_mask = (bio7.get("Whole Root", np.zeros((1, 1), dtype=np.uint8)) > 0)

    if not wr_mask.any():
        # Nothing segmented — return all-NaN
        for k in [
            "root_area_px", "root_perimeter_px", "root_circularity", "root_diameter_px",
            "Epidermis_area_px", "Epidermis_frac",
            "Exodermis_area_px", "Exodermis_frac",
            "Cortex_area_px", "Cortex_frac",
            "Cortex_net_area_px", "Cortex_net_frac",
            "Aerenchyma_area_px", "Aerenchyma_frac", "Aerenchyma_cortex_frac",
            "Endodermis_area_px", "Endodermis_frac",
            "Vascular_area_px", "Vascular_frac",
            "cortex_stele_ratio", "cortex_thickness_px",
            "aerenchyma_pocket_count",
            "aerenchyma_mean_pocket_area_px", "aerenchyma_pocket_area_cv",
            "aerenchyma_inner_frac", "aerenchyma_outer_frac",
        ]:
            rows[k] = np.nan
        return rows

    # Use regionprops on the largest connected component of the whole root
    labeled_wr = label(wr_mask)
    props = regionprops(labeled_wr)
    main = max(props, key=lambda p: p.area)

    root_area = float(main.area)
    root_perim = float(main.perimeter)
    cy, cx = main.centroid                        # root centroid for later use

    rows["root_area_px"]      = root_area
    rows["root_perimeter_px"] = root_perim
    rows["root_circularity"]  = (
        float(4 * np.pi * root_area / root_perim ** 2)
        if root_perim > 0 else np.nan
    )
    rows["root_diameter_px"]  = float(main.equivalent_diameter)

    # ── Tissue areas ─────────────────────────────────────────────────────────
    tissues = ["Epidermis", "Exodermis", "Cortex", "Aerenchyma", "Endodermis", "Vascular"]
    tissue_areas = {}

    for t in tissues:
        mask = bio7.get(t, np.zeros_like(wr_mask))
        area = float((mask > 0).sum())
        tissue_areas[t] = area
        rows[f"{t}_area_px"] = area
        rows[f"{t}_frac"]    = area / root_area if root_area > 0 else np.nan

    # Net cortex (tissue only — Cortex mask spans InnerExo→OuterEndo including aerenchyma holes)
    cortex_net = tissue_areas["Cortex"] - tissue_areas["Aerenchyma"]
    rows["Cortex_net_area_px"] = max(cortex_net, 0.0)
    rows["Cortex_net_frac"]    = max(cortex_net, 0.0) / root_area if root_area > 0 else np.nan

    # Aerenchyma as fraction of cortex (RCA — Lynch lab canonical metric)
    rows["Aerenchyma_cortex_frac"] = (
        tissue_areas["Aerenchyma"] / tissue_areas["Cortex"]
        if tissue_areas["Cortex"] > 0 else np.nan
    )

    # Cortex:stele ratio (canonical Lynch lab drought-adaptation trait)
    stele  = tissue_areas["Vascular"]
    cortex = tissue_areas["Cortex"]
    rows["cortex_stele_ratio"] = cortex / stele if stele > 0 else np.nan

    # ── Cortex thickness (equivalent-circle approximation) ───────────────────
    # filled_outer_endo ≈ Vascular + Endodermis ring
    # filled_inner_exo  ≈ Vascular + Endodermis + Cortex
    filled_endo_area = tissue_areas["Vascular"] + tissue_areas["Endodermis"]
    filled_exo_inner_area = filled_endo_area + tissue_areas["Cortex"]
    r_endo = np.sqrt(filled_endo_area / np.pi) if filled_endo_area > 0 else 0.0
    r_exo  = np.sqrt(filled_exo_inner_area / np.pi) if filled_exo_inner_area > 0 else 0.0
    rows["cortex_thickness_px"] = float(max(r_exo - r_endo, 0.0))

    # ── Aerenchyma pocket statistics + spatial distribution ───────────────────
    aer_mask    = bio7.get("Aerenchyma", np.zeros_like(wr_mask)) > 0
    cortex_mask = bio7.get("Cortex",     np.zeros_like(wr_mask)) > 0

    if aer_mask.any():
        aer_labeled  = label(aer_mask)
        aer_props    = regionprops(aer_labeled)
        pocket_areas = [float(p.area) for p in aer_props]
        rows["aerenchyma_pocket_count"]        = len(pocket_areas)
        rows["aerenchyma_mean_pocket_area_px"] = float(np.mean(pocket_areas))
        rows["aerenchyma_pocket_area_cv"]      = (
            float(np.std(pocket_areas) / np.mean(pocket_areas))
            if len(pocket_areas) > 1 else 0.0
        )

        # Spatial distribution: inner vs outer cortex half (radial from centroid)
        aer_in_cortex = aer_mask & cortex_mask
        if aer_in_cortex.any() and cortex_mask.any():
            yy, xx = np.indices(wr_mask.shape)
            dist_map = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            cortex_median_dist = float(np.median(dist_map[cortex_mask]))
            aer_dists = dist_map[aer_in_cortex]
            n_inner = int((aer_dists < cortex_median_dist).sum())
            rows["aerenchyma_inner_frac"] = float(n_inner / aer_dists.size)
            rows["aerenchyma_outer_frac"] = float(1.0 - rows["aerenchyma_inner_frac"])
        else:
            rows["aerenchyma_inner_frac"] = np.nan
            rows["aerenchyma_outer_frac"] = np.nan
    else:
        rows["aerenchyma_pocket_count"]        = 0
        rows["aerenchyma_mean_pocket_area_px"] = np.nan
        rows["aerenchyma_pocket_area_cv"]      = np.nan
        rows["aerenchyma_inner_frac"]          = np.nan
        rows["aerenchyma_outer_frac"]          = np.nan

    # ── µm conversions (optional) ─────────────────────────────────────────────
    if um_per_px is not None:
        area_scale = um_per_px ** 2
        for k in list(rows.keys()):
            if k.endswith("_area_px"):
                rows[k.replace("_px", "_um2")] = rows[k] * area_scale
            elif k.endswith("_diameter_px") or k.endswith("_perimeter_px") or k.endswith("_thickness_px"):
                rows[k.replace("_px", "_um")] = rows[k] * um_per_px

    return rows


def measure_barrier_intensity_batch(
    raw,
    bio7,
    threshold_mode,
    threshold_dict,
):
    ch_idx = {"TRITC": 0, "FITC": 1, "DAPI": 2}
    rows = {}

    for tissue in BARRIER_TISSUES:
        mask = _tissue_mask(bio7, tissue)

        for ch in ["TRITC", "FITC"]:
            prefix = f"{tissue}_{ch}"

            img_raw = raw[..., ch_idx[ch]].astype(np.float32)
            img_norm = normalize_channel(img_raw)

            if not mask.any():
                rows[f"{prefix}_raw_mean"] = np.nan
                rows[f"{prefix}_thresholded_raw_mean"] = np.nan
                continue

            raw_vals = img_raw[mask]

            low_thr, high_thr = threshold_dict[(tissue, ch)]

            if threshold_mode == "Raw intensity":
                positive_mask = mask & (img_raw >= low_thr) & (img_raw <= high_thr)
            else:
                positive_mask = mask & (img_norm >= low_thr) & (img_norm <= high_thr)

            positive_vals = img_raw[positive_mask]

            mean_val = float(np.mean(raw_vals))
            rows[f"{prefix}_raw_mean"]            = mean_val
            rows[f"{prefix}_raw_std"]             = float(np.std(raw_vals))
            rows[f"{prefix}_signal_spatial_cv"]   = (
                float(np.std(raw_vals) / mean_val) if mean_val > 0 else np.nan
            )
            rows[f"{prefix}_thresholded_raw_mean"] = (
                float(np.mean(positive_vals)) if positive_vals.size else np.nan
            )
            rows[f"{prefix}_thresholded_frac"] = (
                float(positive_vals.size / raw_vals.size) if raw_vals.size > 0 else np.nan
            )

    # FITC/TRITC ratio per tissue (lignin vs suberin balance)
    for tissue in BARRIER_TISSUES:
        fitc  = rows.get(f"{tissue}_FITC_raw_mean",  np.nan)
        tritc = rows.get(f"{tissue}_TRITC_raw_mean", np.nan)
        rows[f"{tissue}_FITC_TRITC_ratio"] = (
            float(fitc / tritc) if (not np.isnan(fitc) and not np.isnan(tritc) and tritc > 0)
            else np.nan
        )

    return rows


def measure_barrier_completeness(raw, bio7, threshold_dict, threshold_mode, n_angles=360):
    """
    Compute barrier signal completeness for each barrier tissue.

    For each tissue × channel, samples n_angles rays from the root centroid and
    reports the fraction of angular sectors in which above-threshold signal exists
    within the tissue mask.  A value of 1.0 = continuous ring; < 1.0 = gaps
    (e.g. sgn mutants, AMF colonisation zones, incomplete suberin deposition).

    Parameters
    ----------
    raw          : (H, W, C) float32 image array
    bio7         : dict {tissue: (H,W) uint8 mask}
    threshold_dict : {(tissue, channel): (low, high)}
    threshold_mode : "Raw intensity" or "Normalized intensity"
    n_angles     : angular resolution (default 360 = 1° bins)

    Returns
    -------
    dict with keys:
        {tissue}_{ch}_completeness     — fraction of angles with signal [0–1]
        {tissue}_{ch}_completeness_pct — same × 100 for readability
    """
    ch_idx = {"TRITC": 0, "FITC": 1, "DAPI": 2}
    rows = {}

    # Root centroid
    wr_mask = bio7.get("Whole Root", np.zeros(raw.shape[:2], dtype=np.uint8)) > 0
    if not wr_mask.any():
        return rows

    labeled_wr = label(wr_mask)
    props = regionprops(labeled_wr)
    main = max(props, key=lambda p: p.area)
    cy, cx = main.centroid

    h, w = raw.shape[:2]
    yy, xx = np.indices((h, w))
    # Angle of every pixel relative to centroid (-π … π)
    angle_map = np.arctan2(yy - cy, xx - cx)
    angle_bins = np.linspace(-np.pi, np.pi, n_angles + 1)

    for tissue in BARRIER_TISSUES:
        tissue_mask = _tissue_mask(bio7, tissue) if tissue in bio7 else np.zeros((h, w), dtype=bool)

        for ch in ["FITC", "TRITC"]:
            key = f"{tissue}_{ch}"
            if not tissue_mask.any():
                rows[f"{key}_completeness"]     = np.nan
                rows[f"{key}_completeness_pct"] = np.nan
                continue

            img_raw  = raw[..., ch_idx[ch]].astype(np.float32)
            img_norm = normalize_channel(img_raw)

            low_thr, high_thr = threshold_dict.get((tissue, ch), (0.0, np.inf))
            if threshold_mode == "Raw intensity":
                signal_mask = tissue_mask & (img_raw >= low_thr) & (img_raw <= high_thr)
            else:
                signal_mask = tissue_mask & (img_norm >= low_thr) & (img_norm <= high_thr)

            # Vectorised: histogram of angles where signal exists
            signal_angles = angle_map[signal_mask]
            hist, _ = np.histogram(signal_angles, bins=angle_bins)
            completeness = float((hist > 0).sum() / n_angles)

            rows[f"{key}_completeness"]     = completeness
            rows[f"{key}_completeness_pct"] = completeness * 100.0

    return rows

import numpy as np
import cv2
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import ndimage as ndi
from scipy.signal import find_peaks, peak_widths
from skimage import filters

from .scale_utils import compute_root_scale, scaled_ring_defaults


COUNT_PRESETS = {
    "Conservative": {
        "min_spacing_ratio_trigger": 0.50,
        "spacing_cv_trigger": 0.20,
        "weak_peak_trigger": 0.04,
        "severity_threshold": 3,
    },
    "Balanced": {
        "min_spacing_ratio_trigger": 0.45,
        "spacing_cv_trigger": 0.22,
        "weak_peak_trigger": 0.05,
        "severity_threshold": 3,
    },
    "Sensitive": {
        "min_spacing_ratio_trigger": 0.40,
        "spacing_cv_trigger": 0.25,
        "weak_peak_trigger": 0.08,
        "severity_threshold": 4,
    },
}

RING_RANGE_PRESETS = {
    "Inner 0.25–0.65": (0.25, 0.65),
    "Balanced 0.35–0.75": (0.35, 0.75),
    "Outer 0.40–0.80": (0.40, 0.80),
    "Very outer 0.50–0.90": (0.50, 0.90),
}

def spacing_sanity_collapse_peaks(
    signal_smooth,
    peaks,
    min_spacing_ratio_trigger=0.45,
    spacing_cv_trigger=0.22,
    weak_peak_trigger=0.05,
    severity_threshold=3,
    max_iter=5,
):
    peaks = np.array(sorted(set(peaks)), dtype=int)
    n_signal = len(signal_smooth)

    info = {
        "applied": False,
        "raw_count": int(len(peaks)),
        "removed": 0,
        "removed_peaks": [],
        "removed_peak_values": [],
        "removed_peak_spacings": [],
        "reason": "",
    }

    if len(peaks) < 5:
        return peaks, info

    for _ in range(max_iter):
        peaks = np.array(sorted(set(peaks)), dtype=int)

        spacings = np.diff(np.r_[peaks, peaks[0] + n_signal])
        median_spacing = float(np.median(spacings))

        if median_spacing <= 0:
            break

        peak_values = signal_smooth[peaks]
        median_peak_signal = float(np.median(peak_values))

        spacing_cv = float(np.std(spacings) / (np.mean(spacings) + 1e-6))
        weak_peak_fraction = float(np.mean(peak_values < 0.5 * median_peak_signal))
        min_to_median_spacing = float(np.min(spacings) / (median_spacing + 1e-6))
        max_to_median_spacing = float(np.max(spacings) / (median_spacing + 1e-6))

        severity = 0

        # NEW: detect local clustering
        small_spacing_ratio = spacings / (median_spacing + 1e-6)
        n_very_small = np.sum(small_spacing_ratio < 0.5)

        if n_very_small >= 2:
            severity += 1

        if spacing_cv > spacing_cv_trigger:
            severity += 1
        if spacing_cv > spacing_cv_trigger + 0.03:
            severity += 1

        if weak_peak_fraction > weak_peak_trigger:
            severity += 1
        if weak_peak_fraction > weak_peak_trigger + 0.03:
            severity += 1

        if min_to_median_spacing < min_spacing_ratio_trigger:
            severity += 1
        if min_to_median_spacing < min_spacing_ratio_trigger - 0.10:
            severity += 1

        if max_to_median_spacing > 1.5:
            severity += 1

        if severity == 0:
            break

        close_pair_indices = np.where(
            spacings < min_spacing_ratio_trigger * median_spacing
        )[0]

        if len(close_pair_indices) == 0:
            break

        close_pair_indices = sorted(
            close_pair_indices,
            key=lambda i: spacings[i],
        )

        if severity >= severity_threshold:
            n_remove = min(severity, max(2, len(peaks) // 8))
        else:
            n_remove = min(severity, max(1, len(peaks) // 10))

        removed_this_round = 0
        remove = set()

        for i in close_pair_indices:
            if removed_this_round >= n_remove:
                break

            p1 = int(peaks[i])
            p2 = int(peaks[(i + 1) % len(peaks)])

            if p1 in remove or p2 in remove:
                continue

            v1 = float(signal_smooth[p1])
            v2 = float(signal_smooth[p2])

            remove_peak = p1 if v1 < v2 else p2

            # log BEFORE removal
            info["removed_peaks"].append(int(remove_peak))
            info["removed_peak_values"].append(float(signal_smooth[remove_peak]))

            # approximate spacing context (use current spacing)
            spacing_val = float(spacings[i])
            info["removed_peak_spacings"].append(spacing_val)

            remove.add(remove_peak)
            removed_this_round += 1

        if not remove:
            break

        peaks = np.array([p for p in peaks if int(p) not in remove], dtype=int)

        info["applied"] = True
        info["removed"] += len(remove)
        info["reason"] = (
            f"spacing_cv={spacing_cv:.3f}; "
            f"weak_peak_fraction={weak_peak_fraction:.3f}; "
            f"min_to_median_spacing={min_to_median_spacing:.3f}; "
            f"max_to_median_spacing={max_to_median_spacing:.3f}; "
            f"severity={severity}; "
            f"removed_this_round={len(remove)}"
        )

    # -------------------------
    # Missed-cell recovery (large gap)
    # -------------------------

    # if len(peaks) > 3:
    #     spacings = np.diff(np.r_[peaks, peaks[0] + len(signal_smooth)])
    #     median_spacing = float(np.median(spacings))

    #     if median_spacing > 0:
    #         max_to_median_spacing = float(np.max(spacings) / (median_spacing + 1e-6))

    #         if max_to_median_spacing > 1.8:
    #             largest_gap_idx = int(np.argmax(spacings))

    #             p1 = int(peaks[largest_gap_idx])
    #             p2 = int(peaks[(largest_gap_idx + 1) % len(peaks)])

    #             mid = int((p1 + p2) / 2) % len(signal_smooth)

    #             peaks = np.append(peaks, mid)
    #             peaks = np.sort(peaks)

    #             info["reason"] += f"; gap_recovery_added_peak={mid}"

    return peaks, info


def width_cleanup_peaks(signal_smooth, peaks, max_remove=4):
    """
    Optional cleanup for split peaks.
    Removes narrow peaks from unusually close peak pairs.
    """
    peaks = np.array(sorted(set(peaks)), dtype=int)

    if len(peaks) < 6:
        return peaks, {"applied": False, "removed": 0, "reason": "too few peaks"}

    widths, _, _, _ = peak_widths(signal_smooth, peaks, rel_height=0.5)
    median_width = float(np.median(widths)) if len(widths) else 0

    spacings = np.diff(np.r_[peaks, peaks[0] + len(signal_smooth)])
    median_spacing = float(np.median(spacings)) if len(spacings) else 0

    if median_width <= 0 or median_spacing <= 0:
        return peaks, {"applied": False, "removed": 0, "reason": "invalid width/spacing"}

    remove = set()

    for i, gap in enumerate(spacings):
        if len(remove) >= max_remove:
            break

        if gap >= 0.60 * median_spacing:
            continue

        p1 = int(peaks[i])
        p2 = int(peaks[(i + 1) % len(peaks)])

        w1 = float(widths[i])
        w2 = float(widths[(i + 1) % len(peaks)])
        v1 = float(signal_smooth[p1])
        v2 = float(signal_smooth[p2])

        p1_narrow = w1 < 0.65 * median_width
        p2_narrow = w2 < 0.65 * median_width

        if p1_narrow and not p2_narrow:
            remove.add(p1)
        elif p2_narrow and not p1_narrow:
            remove.add(p2)
        elif p1_narrow and p2_narrow:
            remove.add(p1 if v1 < v2 else p2)

    if not remove:
        return peaks, {"applied": False, "removed": 0, "reason": "no narrow close split peaks"}

    refined = np.array([p for p in peaks if int(p) not in remove], dtype=int)

    if len(refined) < 5:
        return peaks, {"applied": False, "removed": 0, "reason": "would reduce below 5 peaks"}

    return refined, {
        "applied": True,
        "removed": int(len(remove)),
        "reason": (
            f"removed={len(remove)}; "
            f"median_width={median_width:.2f}; "
            f"median_spacing={median_spacing:.2f}; "
            f"removed_peaks={sorted(list(remove))}"
        ),
    }


def radial_beam_layer_count(raw, bio7, channel, n_beams=72, smooth_sigma=2, peak_distance=8, peak_prominence=200):
    """
    Estimate cortex layer number by sampling radial beams through cortex.
    Counts intensity peaks along each beam inside cortex excluding aerenchyma.
    """

    ch_idx = {"TRITC": 0, "FITC": 1, "DAPI": 2}
    img = raw[..., ch_idx[channel]]

    cortex = bio7["Cortex"] > 0
    aer = bio7["Aerenchyma"] > 0
    cortex_mask = cortex & (~aer)

    vascular = bio7["Vascular"] > 0

    # Use vascular centroid as root center proxy
    if vascular.any():
        ys, xs = np.where(vascular)
    else:
        ys, xs = np.where(bio7["Whole Root"] > 0)

    cy = np.mean(ys)
    cx = np.mean(xs)

    h, w = img.shape
    max_r = int(np.sqrt(h**2 + w**2))

    beam_counts = []
    beam_points = []
    peak_points = []

    for angle in np.linspace(0, 2 * np.pi, n_beams, endpoint=False):
        rs = np.arange(0, max_r)
        yy = (cy + rs * np.sin(angle)).astype(int)
        xx = (cx + rs * np.cos(angle)).astype(int)

        valid = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        yy = yy[valid]
        xx = xx[valid]

        in_cortex = cortex_mask[yy, xx]

        if in_cortex.sum() < 10:
            continue

        # keep only cortex positions along this beam
        yy_c = yy[in_cortex]
        xx_c = xx[in_cortex]
        signal = img[yy_c, xx_c].astype(float)

        if len(signal) < 10:
            continue

        signal_smooth = filters.gaussian(signal, sigma=smooth_sigma)

        peaks, _ = find_peaks(
            signal_smooth,
            distance=peak_distance,
            prominence=peak_prominence,
        )

        if len(peaks) == 0:
            continue

        beam_counts.append(len(peaks))
        beam_points.append((yy_c, xx_c))

        for p in peaks:
            peak_points.append((yy_c[p], xx_c[p]))

    if len(beam_counts) == 0:
        return {
            "median_layers": 0,
            "mean_layers": 0,
            "valid_beams": 0,
            "beam_counts": [],
            "beam_points": [],
            "peak_points": [],
        }

    return {
        "median_layers": float(np.median(beam_counts)),
        "mean_layers": float(np.mean(beam_counts)),
        "valid_beams": int(len(beam_counts)),
        "beam_counts": beam_counts,
        "beam_points": beam_points,
        "peak_points": peak_points,
    }

def ring_peak_count(
    raw,
    bio7,
    tissue,
    channel,
    ring_position=0.5,
    ring_width=None,
    n_points=720,
    smooth_sigma=2,
    peak_distance=None,
    peak_prominence=75,
    local_contrast_threshold=150,
    peak_width=None,
    merge_distance=None,
    enable_gap_fill=False,
    enable_dense_check=False,
    enable_wide_peak_split=False,
    enable_close_doublet_collapse=False,
    auto_peak_distance=False,
    enable_adaptive_peak_cleanup=False,
    enable_valley_rescue=False,
    ring_start_offset=0,
    preset_params=None,
    scale=None,
    verbose=False,
    _apply_auto_seam=True,
):

    if preset_params is None:
        preset_params = COUNT_PRESETS["Balanced"]

    # Track whether any scale-dependent params were auto-filled from pixel-unit
    # defaults.  ring_width is consumed in pixel space (radial sampling) and is
    # fine as-is.  peak_distance / merge_distance / peak_width are consumed in
    # signal-index space by find_peaks(), so if they came from scaled_ring_defaults
    # (pixel units) we replace them with adaptive estimates derived from the
    # normalized ring signal after extraction.
    _auto_scaled = any(v is None for v in (ring_width, peak_distance, peak_width, merge_distance))
    if _auto_scaled:
        if scale is None:
            scale = compute_root_scale(bio7)
        _defaults = scaled_ring_defaults(scale)
        if ring_width is None:
            ring_width = _defaults["ring_width"]
        if peak_distance is None:
            peak_distance = _defaults["peak_distance"]
        if peak_width is None:
            peak_width = _defaults["peak_width"]
        if merge_distance is None:
            merge_distance = _defaults["merge_distance"]

    ch_idx = {"TRITC": 0, "FITC": 1, "DAPI": 2}
    img = raw[..., ch_idx[channel]]

    tissue_mask = bio7[tissue] > 0
    vascular = bio7["Vascular"] > 0

    if vascular.any():
        ys, xs = np.where(vascular)
    else:
        ys, xs = np.where(bio7["Whole Root"] > 0)

    cy = np.mean(ys)
    cx = np.mean(xs)

    h, w = img.shape
    max_r = int(np.sqrt(h**2 + w**2))
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    signals = []
    ring_points = []

    for a in angles:
        rs = np.arange(0, max_r)

        yy = np.round(cy + rs * np.sin(a)).astype(int)
        xx = np.round(cx + rs * np.cos(a)).astype(int)

        valid = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        yy = yy[valid]
        xx = xx[valid]
        rs = rs[valid]

        inside = tissue_mask[yy, xx]

        if inside.sum() < 2:
            signals.append(0)
            ring_points.append((int(cy), int(cx)))
            continue

        tissue_rs = rs[inside]
        r_inner = tissue_rs.min()
        r_outer = tissue_rs.max()
        r_mid = r_inner + ring_position * (r_outer - r_inner)

        vals = []
        pts = []

        inner_w = max(2, int(ring_width * 0.8))
        outer_w = max(3, int(ring_width * 1.4))

        for dr in range(-inner_w, outer_w + 1):
            rr = r_mid + dr

            y = int(np.clip(round(cy + rr * np.sin(a)), 0, h - 1))
            x = int(np.clip(round(cx + rr * np.cos(a)), 0, w - 1))

            if tissue_mask[y, x]:
                vals.append(float(img[y, x]))
                pts.append((y, x))

        if vals:
            vals_arr = np.asarray(vals, dtype=float)
            signals.append(float(np.percentile(vals_arr, 85)))
            ring_points.append(pts[len(pts) // 2])
        else:
            y = int(np.clip(round(cy + r_mid * np.sin(a)), 0, h - 1))
            x = int(np.clip(round(cx + r_mid * np.cos(a)), 0, w - 1))
            signals.append(0)
            ring_points.append((y, x))

    signal = np.array(signals, dtype=float)

    signal_smooth = filters.gaussian(signal, sigma=smooth_sigma)

    if enable_adaptive_peak_cleanup:
        rough = signal_smooth - filters.gaussian(signal_smooth, sigma=20)
        rough = rough - np.min(rough)

        rough_p99 = np.percentile(rough, 99)
        if rough_p99 > 0:
            rough = np.clip(rough / rough_p99, 0, 1.5)

        rough_peaks, _ = find_peaks(
            rough,
            distance=max(3, peak_distance),
            prominence=0.05,
            width=1,
        )

        if len(rough_peaks) >= 6:
            rough_spacings = np.diff(np.r_[rough_peaks, rough_peaks[0] + len(rough)])
            rough_spacing_cv = float(np.std(rough_spacings) / (np.mean(rough_spacings) + 1e-6))
            rough_median_spacing = float(np.median(rough_spacings))
            rough_min_ratio = float(np.min(rough_spacings) / (rough_median_spacing + 1e-6))

            if rough_spacing_cv > 0.22 or rough_min_ratio < 0.45:
                signal_smooth = filters.gaussian(
                    signal,
                    sigma=max(float(smooth_sigma), float(smooth_sigma) * 1.5),
                )

    background = filters.gaussian(signal_smooth, sigma=20)
    signal_smooth = signal_smooth - background
    signal_smooth = signal_smooth - np.min(signal_smooth)

    # Normalize ring signal to relative 0–1 scale per image/ring
    p99 = np.percentile(signal_smooth, 99)
    if p99 > 0:
        signal_smooth = np.clip(signal_smooth / p99, 0, 1.5)
    else:
        signal_smooth = np.zeros_like(signal_smooth)

    # Effective prominence threshold on the normalized [0, 1.5] signal.
    # Exodermis walls are thinner / fainter — use a lower threshold.
    if tissue == "Exodermis":
        effective_prominence = 0.025
    else:
        effective_prominence = 0.05

    # ── Adaptive peak_distance (fixes pixel → index unit mismatch) ────────────
    # scaled_ring_defaults() returns pixel-unit values, but scipy find_peaks()
    # expects signal-index units (0 … n_points).  Rather than a geometric
    # pixel→index conversion that requires knowing the ring radius, we estimate
    # the correct peak_distance directly from the normalized ring signal — this
    # is more accurate and automatically adapts to ring density (24 cells for
    # Rice endo vs 80+ for Sorghum exo).
    if _auto_scaled:
        _coarse_pd   = max(3, int(0.005 * len(signal_smooth)))   # ~7 idx for n_points=1440
        _coarse_prom = max(0.02, 0.4 * effective_prominence)
        _coarse_peaks, _ = find_peaks(
            signal_smooth,
            distance=_coarse_pd,
            prominence=_coarse_prom,
            width=1,
        )
        if len(_coarse_peaks) >= 6:
            _coarse_sps = np.diff(
                np.r_[_coarse_peaks, _coarse_peaks[0] + len(signal_smooth)]
            )
            _med_sp = float(np.median(_coarse_sps))
            _good_sps = _coarse_sps[
                (_coarse_sps >= 0.35 * _med_sp) &
                (_coarse_sps <= 2.5  * _med_sp)
            ]
            if len(_good_sps) >= 4:
                _est_sp = float(np.median(_good_sps))
                peak_distance  = max(3, min(
                    round(0.40 * _est_sp),
                    int(0.04 * len(signal_smooth)),
                ))
                merge_distance = max(4, round(0.70 * peak_distance))
                peak_width     = max(1, round(0.30 * peak_distance))

    # Explicit ACF override when caller passes a fixed peak_distance but still
    # wants signal-adaptive refinement. Skipped when _auto_scaled already ran.
    if auto_peak_distance and not _auto_scaled:
        n = len(signal_smooth)
        s = signal_smooth - signal_smooth.mean()
        power = np.abs(np.fft.rfft(s, n=n)) ** 2
        acf = np.fft.irfft(power, n=n)[:n // 2]
        if acf[0] > 0:
            acf /= acf[0]
        min_lag = max(4, int(0.01 * n))
        acf_peaks, _ = find_peaks(acf[min_lag:], prominence=0.05)
        if len(acf_peaks) > 0:
            dominant_period = int(acf_peaks[0]) + min_lag
            peak_distance  = max(3, min(round(0.50 * dominant_period), int(0.04 * n)))
            merge_distance = max(4, round(0.60 * dominant_period))
            peak_width     = max(1, round(0.20 * dominant_period))

    # BUG 9 FIX: Auto-avoid circular seam is now gated by _apply_auto_seam.
    # When called from multi_ring_peak_count, the caller computes the offset once
    # and passes it in ring_start_offset with _apply_auto_seam=False, so each
    # ring sub-call uses the same shared rotation and peaks stay aligned.
    auto_offset = 0
    if _apply_auto_seam:
        seam_window = max(20, int(0.04 * len(signal_smooth)))

        seam_idx = np.r_[
            np.arange(0, seam_window),
            np.arange(len(signal_smooth) - seam_window, len(signal_smooth)),
        ]

        seam_vals = signal_smooth[seam_idx]
        seam_peak_idx = seam_idx[np.argmax(seam_vals)]
        seam_peak_val = signal_smooth[seam_peak_idx]

        global_peak_ref = np.percentile(signal_smooth, 95)

        if seam_peak_val >= 0.75 * global_peak_ref:
            auto_offset = int(len(signal_smooth) // 12)  # ~30 degrees

    ring_start_offset = int(ring_start_offset) + auto_offset
    ring_start_offset = int(ring_start_offset) % len(signal_smooth)

    if ring_start_offset != 0:
        signal_smooth = np.roll(signal_smooth, -ring_start_offset)
        signal = np.roll(signal, -ring_start_offset)
        ring_points = ring_points[ring_start_offset:] + ring_points[:ring_start_offset]

    # -------------------------
    # Circular peak detection
    # -------------------------
    pad = max(peak_distance * 3, 30)

    signal_for_peaks = np.concatenate([
        signal_smooth[-pad:],
        signal_smooth,
        signal_smooth[:pad],
    ])

    raw_peaks_pad, _ = find_peaks(
        signal_for_peaks,
        distance=peak_distance,
        prominence=effective_prominence,
        width=peak_width,
    )

    peaks = raw_peaks_pad - pad
    peaks = peaks[(peaks >= 0) & (peaks < len(signal_smooth))]
    peaks = np.array(sorted(set(peaks)), dtype=int)

    # -------------------------
    # Optional valley rescue for merged peaks
    # -------------------------
    if enable_valley_rescue and len(peaks) >= 5:
        peaks = rescue_peaks_from_valleys(
            signal=signal_smooth,
            peaks=peaks,
            min_gap_factor=1.45,
            min_valley_drop=0.04,
            min_distance=max(3, int(peak_distance)),
        )

    # Scale merge distance to ring length so large roots do not break spacing logic.
    effective_merge_distance = min(
        int(merge_distance),
        max(int(0.02 * len(signal_smooth)), int(peak_distance)),
    )

    # -------------------------
    # Soft merge:
    # merge only weak shoulders, keep close real doublets
    # -------------------------
    if len(peaks) > 1:
        merged = [int(peaks[0])]

        for p in peaks[1:]:
            p = int(p)
            prev = merged[-1]

            dist = p - prev
            if dist < effective_merge_distance:
                prev_val = signal_smooth[prev]
                p_val = signal_smooth[p]

                strength_ratio = min(prev_val, p_val) / (max(prev_val, p_val) + 1e-6)

                # only merge if one peak is clearly a weak shoulder
                if strength_ratio < 0.70:
                    if p_val > prev_val:
                        merged[-1] = p
                else:
                    merged.append(p)
            else:
                merged.append(p)

        peaks = np.array(merged, dtype=int)

    # -------------------------
    # Optional gap-fill rescue
    # -------------------------
    if enable_gap_fill and len(peaks) >= 5:
        circular_peaks = np.array(sorted(peaks))
        spacings = np.diff(np.r_[circular_peaks, circular_peaks[0] + len(signal_smooth)])
        median_spacing = np.median(spacings)

        rescue_peaks = list(peaks)

        for i in range(len(circular_peaks)):
            p1 = circular_peaks[i]
            p2 = circular_peaks[(i + 1) % len(circular_peaks)]
            gap = (p2 - p1) % len(signal_smooth)

            is_large_gap = gap >= 1.6 * median_spacing
            is_medium_gap = gap >= 1.25 * median_spacing

            if not (is_large_gap or is_medium_gap):
                continue

            start = (p1 + int(0.25 * median_spacing)) % len(signal_smooth)
            end = (p2 - int(0.25 * median_spacing)) % len(signal_smooth)

            if start < end:
                gap_idx = np.arange(start, end)
            else:
                gap_idx = np.r_[np.arange(start, len(signal_smooth)), np.arange(0, end)]

            if len(gap_idx) < 3:
                continue

            gap_signal = signal_smooth[gap_idx]

            local_peaks, _ = find_peaks(
                gap_signal,
                distance=max(2, int(0.4 * median_spacing)),
                prominence=max(0.02, 0.25 * effective_prominence),
                width=1,
            )

            if len(local_peaks) == 0:
                continue

            candidate_indices = gap_idx[local_peaks]
            candidate_values = signal_smooth[candidate_indices]
            candidate = int(candidate_indices[np.argmax(candidate_values)])

            rescue_peaks.append(candidate)

        peaks = np.array(sorted(set(rescue_peaks)), dtype=int)

    # -------------------------
    # Optional wide-peak split rescue
    # -------------------------
    if enable_wide_peak_split and len(peaks) >= 5:
        spacings = np.diff(np.r_[peaks, peaks[0] + len(signal_smooth)])
        spacing_cv = float(np.std(spacings) / (np.mean(spacings) + 1e-6))
        max_to_median = float(np.max(spacings) / (np.median(spacings) + 1e-6))

        if not (spacing_cv > 0.30 or max_to_median > 2.2):
            pass  # skip splitting entirely
        else:
            widths, _, _, _ = peak_widths(signal_smooth, peaks, rel_height=0.5)
            median_width = np.median(widths)

            split_peaks = []

            for p, width in zip(peaks, widths):
                if width > 1.15 * median_width:
                    window = max(int(width), int(1.2 * peak_distance))

                    left = max(0, int(p - window))
                    right = min(len(signal_smooth), int(p + window + 1))

                    sub_signal = signal_smooth[left:right]

                    sub_peaks, _ = find_peaks(
                        sub_signal,
                        distance=max(2, int(0.30 * peak_distance)),
                        prominence=max(10, 0.25 * peak_prominence),
                        width=1,
                    )

                    if len(sub_peaks) >= 2:
                        candidate_peaks = left + sub_peaks
                        candidate_vals = signal_smooth[candidate_peaks]
                        top_k = min(2, len(candidate_peaks))
                        keep = candidate_peaks[np.argsort(candidate_vals)[-top_k:]]
                        split_peaks.extend(list(keep))
                    else:
                        split_peaks.append(int(p))
                else:
                    split_peaks.append(int(p))

            peaks = np.array(sorted(set(split_peaks)), dtype=int)

    # -------------------------
    # Final biological filters ONCE
    # -------------------------
    if len(peaks) > 0:
        local_window = max(3, int(0.5 * peak_distance))
        keep_peaks = []

        for p in peaks:
            left = signal_smooth[max(0, p - local_window):p]
            right = signal_smooth[p:min(len(signal_smooth), p + local_window)]

            left_min = np.min(left) if len(left) else signal_smooth[p]
            right_min = np.min(right) if len(right) else signal_smooth[p]

            # Best-valley contrast. This keeps asymmetric but real wall peaks.
            local_min = min(left_min, right_min)
            local_contrast = signal_smooth[p] - local_min

            median_signal = np.median(signal_smooth)

            adaptive_contrast_threshold = max(0.04, 0.12 * median_signal)

            if local_contrast < adaptive_contrast_threshold:
                continue

            # Shape support: reject ultra-thin spike artifacts.
            support_window = max(3, int(0.8 * peak_distance))
            left_i = max(0, p - support_window)
            right_i = min(len(signal_smooth), p + support_window + 1)
            local_region = signal_smooth[left_i:right_i]

            # Reject smudge/spike artifacts:
            # true wall peaks should have local support, not just a needle-like maximum.
            sharp_window = max(2, int(0.25 * peak_distance))
            idx = np.arange(p - sharp_window, p + sharp_window + 1) % len(signal_smooth)
            sharp_region = signal_smooth[idx]

            peak_val = signal_smooth[p]
            shoulder_level = np.percentile(sharp_region, 60)

            artifact_sharpness = peak_val / (shoulder_level + 1e-6)

            if artifact_sharpness > 8.0:
                continue

            if len(local_region) < 3:
                continue

            peak_val = signal_smooth[p]
            local_baseline = np.percentile(local_region, 25)
            half_height = local_baseline + 0.5 * (peak_val - local_baseline)
            support_pixels = np.sum(local_region >= half_height)

            if support_pixels >= max(3, int(peak_width)):
                keep_peaks.append(int(p))

        peaks = np.array(sorted(set(keep_peaks)), dtype=int)

    # Collapse biologically implausibly close double-edge peaks.
    # These are usually two sides of the same thick wall / EFI artifact.
    if enable_close_doublet_collapse and len(peaks) >= 5:
        peaks = np.array(sorted(peaks), dtype=int)

        changed = True
        while changed and len(peaks) >= 5:
            changed = False

            spacings = np.diff(np.r_[peaks, peaks[0] + len(signal_smooth)])
            median_spacing = np.median(spacings)

            # Too-close peaks relative to the typical cell spacing
            close_i = np.where(spacings < 0.45 * median_spacing)[0]

            if len(close_i) == 0:
                break

            remove = set()

            for i in close_i:
                p1 = int(peaks[i])
                p2 = int(peaks[(i + 1) % len(peaks)])

                # keep the stronger of the two
                if signal_smooth[p1] >= signal_smooth[p2]:
                    remove.add(p2)
                else:
                    remove.add(p1)

            new_peaks = np.array([p for p in peaks if int(p) not in remove], dtype=int)

            if len(new_peaks) < len(peaks):
                peaks = new_peaks
                changed = True

    # Conservative large-gap rescue:
    # If final accepted peaks leave an unusually large angular gap,
    # search that gap for one clean missed local maximum.
    if len(peaks) >= 5:
        peaks = np.array(sorted(peaks), dtype=int)
        spacings = np.diff(np.r_[peaks, peaks[0] + len(signal_smooth)])
        median_spacing = np.median(spacings)

        rescue_peaks = list(peaks)

        for i, gap in enumerate(spacings):
            is_large_gap = gap >= 1.6 * median_spacing
            is_medium_gap = gap >= 1.25 * median_spacing

            if not (is_large_gap or is_medium_gap):
                continue

            p1 = peaks[i]
            p2 = peaks[(i + 1) % len(peaks)]

            start = (p1 + int(0.25 * median_spacing)) % len(signal_smooth)
            end = (p2 - int(0.25 * median_spacing)) % len(signal_smooth)

            if start < end:
                gap_idx = np.arange(start, end)
            else:
                gap_idx = np.r_[np.arange(start, len(signal_smooth)), np.arange(0, end)]

            if len(gap_idx) < 5:
                continue

            gap_signal = signal_smooth[gap_idx]

            local_peaks, _ = find_peaks(
                gap_signal,
                distance=max(2, int(0.35 * median_spacing)),
                prominence=max(0.03, 0.25 * effective_prominence),
                width=1,
            )

            if len(local_peaks) == 0:
                continue

            candidate_indices = gap_idx[local_peaks]
            candidate_values = signal_smooth[candidate_indices]
            candidate = int(candidate_indices[np.argmax(candidate_values)])

            # Require real local contrast, but slightly relaxed because this is a rescue.
            local_window = max(3, int(0.5 * peak_distance))
            idx = np.arange(candidate - local_window, candidate + local_window + 1) % len(signal_smooth)
            local_region = signal_smooth[idx]

            candidate_contrast = signal_smooth[candidate] - np.min(local_region)

            median_peak_signal = np.median(signal_smooth[peaks])

            # Width similarity rescue:
            # true missed wall peaks should have a width similar to accepted high-confidence peaks.
            ref_widths, _, _, _ = peak_widths(signal_smooth, peaks, rel_height=0.5)
            candidate_widths, _, _, _ = peak_widths(signal_smooth, [candidate], rel_height=0.5)

            median_ref_width = np.median(ref_widths)
            candidate_width = candidate_widths[0]

            width_ok = (
                candidate_width >= 0.35 * median_ref_width and
                candidate_width <= 3.0 * median_ref_width
            )

            if (
                candidate_contrast >= 0.04
                and signal_smooth[candidate] >= 0.25 * median_peak_signal
                and width_ok
            ):
                rescue_peaks.append(candidate)

        peaks = np.array(sorted(set(rescue_peaks)), dtype=int)

    # -------------------------
    # Final spacing sanity collapse
    # -------------------------
    peaks, spacing_sanity_info = spacing_sanity_collapse_peaks(
        signal_smooth=signal_smooth,
        peaks=peaks,
        min_spacing_ratio_trigger=preset_params["min_spacing_ratio_trigger"],
        spacing_cv_trigger=preset_params["spacing_cv_trigger"],
        weak_peak_trigger=preset_params["weak_peak_trigger"],
        severity_threshold=preset_params["severity_threshold"],
        max_iter=5,
    )

    width_cleanup_info = {
        "applied": False,
        "removed": 0,
        "reason": "",
    }

    if enable_adaptive_peak_cleanup and len(peaks) >= 3:
        spacings = np.diff(np.r_[peaks, peaks[0] + len(signal_smooth)])
        spacing_cv = float(np.std(spacings) / (np.mean(spacings) + 1e-6))

        # ONLY apply cleanup if ring is clearly irregular
        if spacing_cv > 0.18:
            peaks, width_cleanup_info = width_cleanup_peaks(
                signal_smooth=signal_smooth,
                peaks=peaks,
                max_remove=4,
            )

    peak_points = [ring_points[p] for p in peaks]
    qc_reason = ""

    # -------------------------
    # QC metrics
    # -------------------------
    if len(peaks) >= 3:
         # QC: detect broad peaks (merged cell walls)
        widths, _, _, _ = peak_widths(signal_smooth, peaks, rel_height=0.5)

        median_width = float(np.median(widths)) if len(widths) else 0

        wide_peak_fraction = (
            float(np.mean(widths > 1.8 * median_width))
            if median_width > 0 else 0
        )

        has_unresolved_wide_peak = bool(
            median_width > 0 and np.any(widths > 2.2 * median_width)
        )

        spacings = np.diff(np.r_[peaks, peaks[0] + len(signal_smooth)])
        mean_spacing = float(np.mean(spacings))
        spacing_cv = float(np.std(spacings) / (mean_spacing + 1e-6))

        peak_values = signal_smooth[peaks]
        median_peak_signal = float(np.median(peak_values))
        min_peak_signal = float(np.min(peak_values))

        weak_peak_fraction = float(np.mean(
            peak_values < 0.5 * median_peak_signal
        ))

        min_spacing = float(np.min(spacings))
        max_spacing = float(np.max(spacings))
        median_spacing = float(np.median(spacings))
        max_to_median_spacing = max_spacing / (median_spacing + 1e-6)
        min_to_median_spacing = min_spacing / (median_spacing + 1e-6)

        peak_cv = float(np.std(peak_values) / (np.mean(peak_values) + 1e-6))

        qc_reasons = []

        if spacing_cv > 0.25:
            qc_reasons.append("high_spacing_cv")

        if max_to_median_spacing > 1.60:
            qc_reasons.append("large_gap")

        if min_to_median_spacing < 0.45:
            qc_reasons.append("small_gap")

        if weak_peak_fraction > 0.25:
            qc_reasons.append("many_weak_peaks")

        if peak_cv > 0.40:
            qc_reasons.append("high_peak_cv")

        if has_unresolved_wide_peak:
            qc_reasons.append("unresolved_wide_peak")

        if len(peaks) < 5:
            qc_reasons.append("too_few_peaks")

        review_needed = len(qc_reasons) > 0
        qc_reason = ";".join(qc_reasons)

        likely_missed_cell = (
            (max_to_median_spacing > 1.45 and spacing_cv > 0.20)
            or has_unresolved_wide_peak
        )

        if review_needed:
            confidence = "low"
        elif spacing_cv > 0.15 or weak_peak_fraction > 0.10 or peak_cv > 0.22:
            confidence = "medium"
        else:
            confidence = "high"

    else:
        mean_spacing = 0
        spacing_cv = 999
        median_peak_signal = 0
        min_peak_signal = 0
        weak_peak_fraction = 1
        review_needed = True
        confidence = "low"

        min_spacing = 0
        max_spacing = 0
        median_spacing = 0
        max_to_median_spacing = 999
        min_to_median_spacing = 0
        peak_cv = 999

        median_width = 0
        wide_peak_fraction = 0
        has_unresolved_wide_peak = True

    if verbose and tissue == "Endodermis":
        print("\n--- RING DEBUG ---")
        print("channel:", channel)
        print("final peaks:", peaks)
        print("count:", len(peaks))

        # show the strongest local maxima near the end of the signal
        end_region = np.arange(max(0, len(signal_smooth) - 180), len(signal_smooth))
        local_peaks_end, _ = find_peaks(
            signal_smooth[end_region],
            distance=2,
            prominence=20,
            width=1,
        )

        local_peaks_end = end_region[local_peaks_end]

        print("candidate-like peaks near signal end:")
        for p in local_peaks_end:
            print(
                p,
                "signal=", round(float(signal_smooth[p]), 1),
                "kept=", int(p) in set(peaks),
            )

    return {
        "count": int(len(peaks)),
        "raw_count_before_spacing_sanity": int(spacing_sanity_info["raw_count"]),
        "spacing_sanity_applied": bool(spacing_sanity_info["applied"]),
        "spacing_sanity_removed": int(spacing_sanity_info["removed"]),
        "spacing_sanity_removed_peaks": spacing_sanity_info.get("removed_peaks", []),
        "spacing_sanity_removed_values": spacing_sanity_info.get("removed_peak_values", []),
        "spacing_sanity_removed_spacings": spacing_sanity_info.get("removed_peak_spacings", []),
        "spacing_sanity_reason": spacing_sanity_info["reason"],
        "adaptive_peak_cleanup_applied": bool(width_cleanup_info["applied"]),
        "adaptive_peak_cleanup_removed": int(width_cleanup_info["removed"]),
        "adaptive_peak_cleanup_reason": width_cleanup_info["reason"],
        "count_status": "spacing_sanity_corrected" if spacing_sanity_info["applied"] else "",
        "correction_applied": -int(spacing_sanity_info["removed"]) - int(width_cleanup_info["removed"]),
        "channel": channel,
        "base_prominence": float(peak_prominence),
        "adaptive_prominence": float(peak_prominence),
        "signal": signal,
        "signal_smooth": signal_smooth,
        "peaks": peaks,
        "ring_points": ring_points,
        "peak_points": peak_points,
        "mean_spacing": mean_spacing,
        "spacing_cv": spacing_cv,
        "median_peak_signal": median_peak_signal,
        "min_peak_signal": min_peak_signal,
        "weak_peak_fraction": weak_peak_fraction,
        "review_needed": review_needed,
        "qc_reason": qc_reason,
        "confidence": confidence,
        "min_spacing": min_spacing,
        "max_spacing": max_spacing,
        "median_spacing": median_spacing,
        "max_to_median_spacing": max_to_median_spacing,
        "min_to_median_spacing": min_to_median_spacing,
        "peak_cv": peak_cv,
        "median_peak_width": median_width,
        "wide_peak_fraction": wide_peak_fraction,
        "has_unresolved_wide_peak": has_unresolved_wide_peak,
        "_effective_start_offset": ring_start_offset,
    }


def rescue_peaks_from_valleys(
    signal,
    peaks,
    min_gap_factor=1.65,
    min_valley_drop=0.08,
    min_distance=8,
):
    """
    Rescue missing peaks inside unusually large gaps between detected peaks.

    signal: 1D ring intensity profile
    peaks: array/list of detected peak indices
    """

    import numpy as np
    from scipy.signal import find_peaks

    signal = np.asarray(signal, dtype=float)
    peaks = np.asarray(sorted(peaks), dtype=int)

    if len(peaks) < 4:
        return peaks

    n = len(signal)

    # circular gaps between existing peaks
    gaps = np.diff(np.r_[peaks, peaks[0] + n])
    median_gap = np.median(gaps)

    if median_gap <= 0:
        return peaks

    rescued = []

    for i, gap in enumerate(gaps):
        if gap < min_gap_factor * median_gap:
            continue

        left = peaks[i]
        right = peaks[(i + 1) % len(peaks)]

        if right <= left:
            segment_idx = np.r_[np.arange(left, n), np.arange(0, right)]
        else:
            segment_idx = np.arange(left, right)

        if len(segment_idx) < 2 * min_distance:
            continue

        seg = signal[segment_idx]

        # find candidate internal peaks inside the large gap
        internal_peaks, props = find_peaks(
            seg,
            distance=min_distance,
        )

        # exclude boundary peaks very close to already accepted peaks
        internal_peaks = internal_peaks[
            (internal_peaks > min_distance) &
            (internal_peaks < len(seg) - min_distance)
        ]

        if len(internal_peaks) == 0:
            continue

        # choose strongest internal peak
        best_local = internal_peaks[np.argmax(seg[internal_peaks])]
        best_global = segment_idx[best_local] % n

        # require some valley separation from both sides
        left_valley = np.min(seg[:best_local])
        right_valley = np.min(seg[best_local:])

        peak_height = seg[best_local]
        local_baseline = max(left_valley, right_valley)

        if (peak_height - local_baseline) >= min_valley_drop:
            rescued.append(best_global)

    if len(rescued) == 0:
        return peaks

    all_peaks = np.asarray(sorted(set(peaks.tolist() + rescued)), dtype=int)

    return all_peaks


def multi_ring_peak_count(
    raw,
    bio7,
    tissue,
    channel,
    ring_position_min=0.25,
    ring_position_max=0.75,
    ring_position_iterations=7,
    ring_consensus_min=3,
    ring_width=None,
    n_points=1440,
    smooth_sigma=2,
    peak_distance=None,
    peak_prominence=60,
    local_contrast_threshold=250,
    peak_width=None,
    merge_distance=None,
    enable_gap_fill=False,
    enable_dense_check=False,
    enable_wide_peak_split=False,
    ring_start_offset=0,
    enable_close_doublet_collapse=True,
    auto_peak_distance=False,
    enable_adaptive_peak_cleanup=False,
    preset_params=None,
    scale=None,
    enable_spacing_rescue=False,
    enable_valley_rescue=False,
    dapi_validation=False,
    dapi_threshold_fraction=0.25,
    consensus_nms_radius=0,
    consensus_nms_fraction=0.0,
    consensus_nms_max_radius=None,
    ):

    if preset_params is None:
        preset_params = COUNT_PRESETS["Balanced"]

    # ring_width is a radial-sampling parameter (pixel units) — auto-fill from
    # root geometry once and share across all ring calls.
    # peak_distance / merge_distance / peak_width are signal-index params whose
    # correct value depends on ring density (cells per ring), which varies
    # widely across species/tissues.  We leave them as None so each
    # ring_peak_count call estimates them adaptively from its own ring signal.
    if scale is None:
        scale = compute_root_scale(bio7)
    if ring_width is None:
        ring_width = scaled_ring_defaults(scale)["ring_width"]
    # Force adaptive estimation inside ring_peak_count regardless of what the
    # caller passed for these three params.
    peak_distance  = None
    merge_distance = None
    peak_width     = None

    positions = np.linspace(
        ring_position_min,
        ring_position_max,
        ring_position_iterations,
    )

    mid_pos = float(positions[len(positions) // 2])
    _seam_probe = ring_peak_count(
        raw=raw,
        bio7=bio7,
        tissue=tissue,
        channel=channel,
        preset_params=preset_params,
        ring_position=mid_pos,
        ring_width=ring_width,
        n_points=n_points,
        smooth_sigma=smooth_sigma,
        peak_distance=peak_distance,       # None → adaptive inside
        peak_prominence=peak_prominence,
        local_contrast_threshold=local_contrast_threshold,
        peak_width=peak_width,             # None → adaptive inside
        merge_distance=merge_distance,     # None → adaptive inside
        enable_gap_fill=False,
        enable_dense_check=False,
        enable_wide_peak_split=False,
        enable_close_doublet_collapse=False,
        auto_peak_distance=auto_peak_distance,
        enable_adaptive_peak_cleanup=enable_adaptive_peak_cleanup,
        enable_valley_rescue=False,
        ring_start_offset=ring_start_offset,
        scale=scale,
        _apply_auto_seam=True,
    )
    shared_start_offset = int(_seam_probe.get("_effective_start_offset", ring_start_offset))

    all_stats = []

    for pos in positions:
        stats = ring_peak_count(
            raw=raw,
            bio7=bio7,
            tissue=tissue,
            channel=channel,
            preset_params=preset_params,
            ring_position=pos,
            ring_width=ring_width,
            n_points=n_points,
            smooth_sigma=smooth_sigma,
            peak_distance=peak_distance,       # None → adaptive inside
            peak_prominence=peak_prominence,
            local_contrast_threshold=local_contrast_threshold,
            peak_width=peak_width,             # None → adaptive inside
            merge_distance=merge_distance,     # None → adaptive inside
            enable_gap_fill=enable_gap_fill,
            enable_dense_check=enable_dense_check,
            enable_wide_peak_split=enable_wide_peak_split,
            enable_close_doublet_collapse=enable_close_doublet_collapse,
            auto_peak_distance=auto_peak_distance,
            enable_adaptive_peak_cleanup=enable_adaptive_peak_cleanup,
            enable_valley_rescue=enable_valley_rescue,
            ring_start_offset=shared_start_offset,
            scale=scale,
            _apply_auto_seam=False,
        )

        stats["ring_position"] = float(pos)
        all_stats.append(stats)

    # collect all peak indices across rings
    peak_entries = []
    for ring_id, stats in enumerate(all_stats):
        for p in stats["peaks"]:
            peak_entries.append({
                "ring_id": ring_id,
                "peak_index": int(p),
                "value": float(stats["signal_smooth"][p]),
                "point": stats["ring_points"][p],
            })

    if len(peak_entries) == 0:
        base = all_stats[len(all_stats) // 2]
        base["count"] = 0
        base["peaks"] = np.array([], dtype=int)
        base["peak_points"] = []
        base["multi_ring_positions"] = positions
        base["multi_ring_support"] = []
        base["confidence"] = "low"
        base["review_needed"] = True
        return base

    all_indices = sorted(set(e["peak_index"] for e in peak_entries))
    _default_cw = int(0.015 * n_points)   # ~22 idx for n_points=1440

    # Consensus window strategy is tissue-dependent:
    #
    # Exodermis: cells are dense (~100 per ring, spacing ~14 idx). The default
    #   window of 22 idx > spacing/2, so adjacent cells merge → severe under-
    #   count. Use center-ring spacing × 0.49 to stay below spacing/2.
    #
    # Endodermis / other: cells are sparse (~20–45 per ring, spacing ~33–70 idx).
    #   The default window of 22 is well below spacing/2 and above typical cross-
    #   ring scatter (±5–10 idx). Any attempt to make the window adaptive for
    #   endo causes regression on samples with noisy ring signals (some rings
    #   return inflated peak counts that corrupt the spacing estimate, leading
    #   to over-small windows and over-counting). Keep default = 22.
    #
    # Note: crown-root endodermis (~50 cells, spacing ~29) sits in a borderline
    #   regime where window=22 occasionally bridges adjacent cells in tightly
    #   packed regions, causing mild undercounting (~2–5 cells). This is a known
    #   limitation; the algorithm was tuned for primary/seminal root anatomy.
    if tissue == "Exodermis":
        _ctr_peaks = sorted(all_stats[len(all_stats) // 2]["peaks"])
        if len(_ctr_peaks) >= 6:
            _ctr_sps = np.diff(np.r_[_ctr_peaks, _ctr_peaks[0] + n_points])
            _ctr_sps = _ctr_sps[_ctr_sps > 2]
            if len(_ctr_sps) >= 4:
                est_spacing = float(np.median(_ctr_sps))
                # 49% of spacing keeps window just below merge threshold
                consensus_window = max(4, min(_default_cw, int(0.49 * est_spacing)))
            else:
                consensus_window = _default_cw
        else:
            consensus_window = _default_cw
    else:
        consensus_window = _default_cw

    used = np.zeros(len(peak_entries), dtype=bool)
    clusters = []

    for i, entry in enumerate(peak_entries):
        if used[i]:
            continue

        cluster = [entry]
        used[i] = True

        for j in range(i + 1, len(peak_entries)):
            if used[j]:
                continue

            d = abs(entry["peak_index"] - peak_entries[j]["peak_index"])
            d = min(d, n_points - d)

            if d <= consensus_window:
                cluster.append(peak_entries[j])
                used[j] = True

        clusters.append(cluster)

    kept_clusters = []

    for cluster in clusters:
        supporting_rings = sorted(set(e["ring_id"] for e in cluster))

        if len(supporting_rings) >= ring_consensus_min:
            kept_clusters.append(cluster)

    # use middle ring for display path/signal
    base = all_stats[len(all_stats) // 2]

    consensus_peaks = []
    consensus_points = []
    support_counts = []

    for cluster in kept_clusters:
        # representative = strongest peak in the cluster
        rep = max(cluster, key=lambda e: e["value"])
        consensus_peaks.append(rep["peak_index"])
        consensus_points.append(rep["point"])
        support_counts.append(len(set(e["ring_id"] for e in cluster)))

    order = np.argsort(consensus_peaks)
    consensus_peaks = np.array(consensus_peaks, dtype=int)[order]
    consensus_points = [consensus_points[i] for i in order]
    support_counts = [support_counts[i] for i in order]

    # --- Local angular support filter ---
    # Keep well-supported peaks even if they are isolated in angular space.
    # Real missing-gap peaks can look isolated, so do not remove them only because
    # they lack close neighbors.
    filtered_peaks = []
    filtered_points = []
    filtered_support = []

    window = int(0.04 * n_points)

    for i, p in enumerate(consensus_peaks):
        d = np.minimum(
            np.abs(consensus_peaks - p),
            n_points - np.abs(consensus_peaks - p),
        )
        neighbors = np.sum(d <= window)

        enough_neighbors = neighbors >= 2
        enough_ring_support = support_counts[i] >= ring_consensus_min

        if enough_neighbors or enough_ring_support:
            filtered_peaks.append(p)
            filtered_points.append(consensus_points[i])
            filtered_support.append(support_counts[i])

    consensus_peaks = np.array(filtered_peaks, dtype=int)
    consensus_points = filtered_points
    support_counts = filtered_support

    # ── Post-consensus NMS: remove satellite peaks from signal splitting ──────
    # When a fluorescent residue (e.g. spectral bleed-through) sits adjacent to a
    # real cell wall, the smoothed signal develops a secondary local maximum
    # right next to the genuine peak.  Both survive multi-ring consensus because
    # the residue is a physical object (full support, normal amplitude).  Their
    # only distinguishing feature is angular proximity: two peaks within
    # the NMS radius steps must be the same physical object.
    #
    # Selection rule: keep the peak with higher ring-support; break ties by
    # signal amplitude.
    #
    # Two modes:
    #   consensus_nms_radius > 0  — fixed radius (caller supplies it in steps)
    #   consensus_nms_fraction > 0 — adaptive: radius = fraction × median_spacing
    #     computed from the pre-NMS peak set.  Scales automatically with cell
    #     density, so it's safe across tissues with different packing (e.g.
    #     primary roots ~50-step spacing vs crown roots ~30-step spacing).
    #     Recommended fraction: ~0.60  (catches satellite pairs at Δ≈0.45×spacing
    #     while leaving genuine adjacent cells at Δ≥0.65×spacing untouched).
    _nms_r = 0
    if len(consensus_peaks) >= 2:
        if consensus_nms_fraction > 0.0:
            _spacings_nms = np.diff(np.r_[
                np.sort(consensus_peaks),
                np.sort(consensus_peaks)[0] + n_points,
            ])
            _med_sp = float(np.median(_spacings_nms))
            _nms_r  = max(1, round(consensus_nms_fraction * _med_sp))
            if consensus_nms_max_radius is not None:
                _nms_r = min(_nms_r, int(consensus_nms_max_radius))
        elif consensus_nms_radius > 0:
            _nms_r = int(consensus_nms_radius)

    base["consensus_nms_radius_used"] = _nms_r

    if _nms_r > 0 and len(consensus_peaks) >= 2:
        _sig = base["signal_smooth"]
        # Greedy NMS: iterate from strongest support → weakest, suppress neighbours
        _order = sorted(range(len(consensus_peaks)),
                        key=lambda i: (support_counts[i], _sig[consensus_peaks[i]]),
                        reverse=True)
        _keep = np.ones(len(consensus_peaks), dtype=bool)
        for _i in _order:
            if not _keep[_i]:
                continue
            _p = consensus_peaks[_i]
            for _j in _order:
                if _i == _j or not _keep[_j]:
                    continue
                _d = abs(int(consensus_peaks[_j]) - int(_p))
                _d = min(_d, n_points - _d)
                if _d <= _nms_r:
                    _keep[_j] = False  # suppress the weaker neighbour

        _ki = [i for i in range(len(consensus_peaks)) if _keep[i]]
        base["consensus_nms_removed"] = int(len(consensus_peaks)) - len(_ki)
        consensus_peaks  = consensus_peaks[_ki]
        consensus_points = [consensus_points[i] for i in _ki]
        support_counts   = [support_counts[i]   for i in _ki]
    else:
        base["consensus_nms_removed"] = 0

    # Final weak-artifact filter on multi-ring consensus: currently disabled
    if False and len(consensus_peaks) >= 5:
        signal_smooth = base["signal_smooth"]
        peak_values = signal_smooth[consensus_peaks]
        median_peak_value = np.median(peak_values)

        spacings = np.diff(np.r_[consensus_peaks, consensus_peaks[0] + len(signal_smooth)])
        median_spacing = np.median(spacings)

        keep = []

        for i, p in enumerate(consensus_peaks):
            left_gap = spacings[i - 1]
            right_gap = spacings[i]

            weak_peak = signal_smooth[p] < 0.17 * median_peak_value
            not_gap_rescue = left_gap < 1.35 * median_spacing and right_gap < 1.35 * median_spacing

            is_gap_boundary = (
                left_gap > 1.2 * median_spacing or
                right_gap > 1.2 * median_spacing
            )

            if weak_peak and not_gap_rescue and not is_gap_boundary:
                continue

            keep.append(i)

        consensus_peaks = consensus_peaks[keep]
        consensus_points = [consensus_points[i] for i in keep]
        support_counts = [support_counts[i] for i in keep]

    # -------------------------
    # Multi-ring gap rescue with instability guard
    # -------------------------
    if len(consensus_peaks) >= 5:
        consensus_peaks = np.array(sorted(consensus_peaks), dtype=int)
        spacings = np.diff(np.r_[consensus_peaks, consensus_peaks[0] + n_points])
        median_spacing = np.median(spacings)

        signal_smooth = base["signal_smooth"]
        peak_values = signal_smooth[consensus_peaks]

        pre_rescue_spacing_cv = float(
            np.std(spacings) / (np.mean(spacings) + 1e-6)
        )
        pre_rescue_min_to_median = float(
            np.min(spacings) / (median_spacing + 1e-6)
        )
        pre_rescue_peak_cv = float(
            np.std(peak_values) / (np.mean(peak_values) + 1e-6)
        )

        rescue_allowed = not (
            pre_rescue_spacing_cv > 0.20 or
            pre_rescue_min_to_median < 0.50 or
            pre_rescue_peak_cv > 0.45
        )

        rescue_peak_entries = []

        if not rescue_allowed:
            base["rescue_skipped_unstable_ring"] = True

        if rescue_allowed:
            for i, gap in enumerate(spacings):
                is_large_gap = gap >= 1.45 * median_spacing
                is_medium_gap = gap >= 1.15 * median_spacing

                if not (is_large_gap or is_medium_gap):
                    continue

                p1 = int(consensus_peaks[i])
                p2 = int(consensus_peaks[(i + 1) % len(consensus_peaks)])

                gap_candidates = []

                for e in peak_entries:
                    p = int(e["peak_index"])

                    if p1 < p2:
                        in_gap = p1 < p < p2
                    else:
                        in_gap = p > p1 or p < p2

                    if in_gap:
                        gap_candidates.append(e)

                if len(gap_candidates) == 0:
                    continue

                used_gap = np.zeros(len(gap_candidates), dtype=bool)
                candidate_clusters = []

                for a, e1 in enumerate(gap_candidates):
                    if used_gap[a]:
                        continue

                    cluster = [e1]
                    used_gap[a] = True

                    for b, e2 in enumerate(gap_candidates):
                        if used_gap[b]:
                            continue

                        d = abs(int(e1["peak_index"]) - int(e2["peak_index"]))
                        d = min(d, n_points - d)

                        if d <= consensus_window:
                            cluster.append(e2)
                            used_gap[b] = True

                    candidate_clusters.append(cluster)

                good_clusters = [
                    c for c in candidate_clusters
                    if len(set(e["ring_id"] for e in c)) >= 2
                ]

                if len(good_clusters) == 0:
                    gap_idx = (
                        np.arange(p1 + 1, p2)
                        if p1 < p2
                        else np.r_[np.arange(p1 + 1, n_points), np.arange(0, p2)]
                    )

                    if len(gap_idx) < 5:
                        continue

                    gap_signal = signal_smooth[gap_idx]

                    local_peaks, _ = find_peaks(
                        gap_signal,
                        distance=max(2, int(0.18 * median_spacing)),
                        prominence=0.03,
                        width=1,
                    )

                    if len(local_peaks) == 0:
                        continue

                    candidate_indices = gap_idx[local_peaks]
                    candidate_values = signal_smooth[candidate_indices]

                    max_gap_value = float(np.max(candidate_values))
                    strong_candidates = np.where(candidate_values >= 0.65 * max_gap_value)[0]

                    if len(strong_candidates) > 2:
                        strong_candidates = strong_candidates[
                            np.argsort(candidate_values[strong_candidates])[-2:]
                        ]

                    for idx_keep in strong_candidates:
                        candidate = int(candidate_indices[idx_keep])

                        left = (candidate - p1) % n_points
                        right = (p2 - candidate) % n_points

                        spacing_ok = (
                            left > 0.4 * median_spacing and
                            right > 0.4 * median_spacing and
                            left < 1.8 * median_spacing and
                            right < 1.8 * median_spacing
                        )

                        creates_extreme = (
                            min(left, right) < 0.35 * median_spacing or
                            max(left, right) > 2.2 * median_spacing
                        )

                        if spacing_ok and not creates_extreme:
                            rescue_peak_entries.append({
                                "ring_id": len(all_stats) // 2,
                                "peak_index": candidate,
                                "value": float(signal_smooth[candidate]),
                                "point": base["ring_points"][candidate],
                            })

                    continue

                best_cluster = max(
                    good_clusters,
                    key=lambda c: np.max([e["value"] for e in c])
                )

                rep = max(best_cluster, key=lambda e: e["value"])
                candidate = int(rep["peak_index"])

                left_split = (candidate - p1) % n_points
                right_split = (p2 - candidate) % n_points

                balanced_split = (
                    left_split >= 0.35 * median_spacing and
                    right_split >= 0.35 * median_spacing and
                    left_split <= 2.20 * median_spacing and
                    right_split <= 2.20 * median_spacing
                )

                candidate_value_ok = rep["value"] >= 0.35 * np.median([
                    e["value"] for e in peak_entries
                ])

                if is_large_gap and balanced_split and candidate_value_ok:
                    rescue_peak_entries.append(rep)

                elif is_medium_gap:
                    stronger_signal = rep["value"] >= 0.5 * np.median([
                        e["value"] for e in peak_entries
                    ])

                    if balanced_split and stronger_signal:
                        rescue_peak_entries.append(rep)

        for rep in rescue_peak_entries:
            p = int(rep["peak_index"])
            if p not in set(consensus_peaks):
                consensus_peaks = np.append(consensus_peaks, p)
                consensus_points.append(rep["point"])
                support_counts.append(
                    len(set(
                        e["ring_id"]
                        for e in peak_entries
                        if abs(int(e["peak_index"]) - p) <= consensus_window
                    ))
                )

        order = np.argsort(consensus_peaks)
        consensus_peaks = np.array(consensus_peaks, dtype=int)[order]
        consensus_points = [consensus_points[i] for i in order]
        support_counts = [support_counts[i] for i in order]

    # -------------------------
    # Spatial same-wall doublet collapse
    # -------------------------
    # Some slanted walls generate two signal peaks at nearly the same physical location.
    # Collapse only when peaks are close both in ring-index space and image-coordinate space.
    if len(consensus_peaks) >= 5:
        order = np.argsort(consensus_peaks)
        consensus_peaks = np.array(consensus_peaks, dtype=int)[order]
        consensus_points = [consensus_points[i] for i in order]
        support_counts = [support_counts[i] for i in order]

        signal_smooth = base["signal_smooth"]

        changed = True
        while changed and len(consensus_peaks) >= 5:
            changed = False

            spacings = np.diff(np.r_[consensus_peaks, consensus_peaks[0] + n_points])
            median_spacing = float(np.median(spacings))

            pts = np.array(consensus_points, dtype=float)
            xy_dists = []

            for i in range(len(pts)):
                p_a = pts[i]
                p_b = pts[(i + 1) % len(pts)]
                xy_dists.append(np.linalg.norm(p_a - p_b))

            median_xy_dist = float(np.median(xy_dists)) if xy_dists else 0

            remove = set()

            for i, gap in enumerate(spacings):
                j = (i + 1) % len(consensus_peaks)

                p1 = int(consensus_peaks[i])
                p2 = int(consensus_peaks[j])

                pt1 = np.array(consensus_points[i], dtype=float)
                pt2 = np.array(consensus_points[j], dtype=float)

                xy_dist = np.linalg.norm(pt1 - pt2)

                # Only collapse extremely close same-wall doublets.
                # Do NOT collapse normal small adjacent cells.
                index_close = gap < 0.35 * median_spacing

                spatial_close = (
                    xy_dist <= 6.0
                    or (median_xy_dist > 0 and xy_dist < 0.25 * median_xy_dist)
                )

                if not (index_close and spatial_close):
                    continue

                # Same-wall duplicates usually have one weaker member.
                # Keep stronger signal; if similar, keep the one with higher multi-ring support.
                v1 = float(signal_smooth[p1])
                v2 = float(signal_smooth[p2])
                s1 = support_counts[i]
                s2 = support_counts[j]

                if abs(v1 - v2) < 0.15 * max(v1, v2):
                    if s1 >= s2:
                        remove.add(j)
                    else:
                        remove.add(i)
                else:
                    if v1 >= v2:
                        remove.add(j)
                    else:
                        remove.add(i)

            if remove:
                keep_idx = [k for k in range(len(consensus_peaks)) if k not in remove]
                consensus_peaks = consensus_peaks[keep_idx]
                consensus_points = [consensus_points[k] for k in keep_idx]
                support_counts = [support_counts[k] for k in keep_idx]
                changed = True

    # -------------------------
    # Conservative geometry-based large-gap fill
    # -------------------------
    if len(consensus_peaks) >= 5:
        signal_smooth = base["signal_smooth"]
        n_signal = len(signal_smooth)

        consensus_peaks = np.array(sorted(consensus_peaks), dtype=int)

        spacings = np.diff(
            np.r_[consensus_peaks, consensus_peaks[0] + n_signal]
        )

        median_spacing = float(np.median(spacings))

        # --- NEW: prevent overcount in compressed rings ---
        min_gap_ratio = float(np.min(spacings) / (median_spacing + 1e-6))

        skip_geometry_fill = False

        if min_gap_ratio < 0.55:
            base["rescue_skipped_compressed_ring"] = True
            skip_geometry_fill = True

        # --- spatial gap computation ---
        pts = np.array(consensus_points, dtype=float)

        spatial_gaps = []
        for j in range(len(pts)):
            a = pts[j]
            b = pts[(j + 1) % len(pts)]
            spatial_gaps.append(np.linalg.norm(a - b))

        median_spatial_gap = float(np.median(spatial_gaps)) if spatial_gaps else 0

        added_peaks = []

        median_peak_signal = float(np.median(signal_smooth[consensus_peaks]))

        if not skip_geometry_fill:
            for i, gap in enumerate(spacings):

                pt1 = pts[i]
                pt2 = pts[(i + 1) % len(pts)]
                spatial_gap = np.linalg.norm(pt1 - pt2)

                large_index_gap = gap > 1.7 * median_spacing
                large_spatial_gap = spatial_gap > 1.6 * median_spatial_gap

                if not (large_index_gap or large_spatial_gap):
                    continue

                p1 = int(consensus_peaks[i])
                p2 = int(consensus_peaks[(i + 1) % len(consensus_peaks)])

                n_missing = int(round(gap / median_spacing)) - 1
                n_missing = max(1, min(n_missing, 5))

                current_min_gap_ratio = float(np.min(spacings) / (median_spacing + 1e-6))
                current_spacing_cv = float(np.std(spacings) / (np.mean(spacings) + 1e-6))

                allow_multi_rescue = (
                    gap > 2.2 * median_spacing and
                    current_min_gap_ratio > 0.55 and
                    current_spacing_cv < 0.25
                )

                if n_missing > 1 and not allow_multi_rescue:
                    continue

                for k in range(1, n_missing + 1):
                    expected = (p1 + int(k * gap / (n_missing + 1))) % n_signal

                    search_radius = int(0.20 * median_spacing)

                    search_idx = np.arange(
                        expected - search_radius,
                        expected + search_radius + 1
                    ) % n_signal

                    candidate = int(search_idx[np.argmax(signal_smooth[search_idx])])

                    left = (candidate - p1) % n_signal
                    right = (p2 - candidate) % n_signal

                    candidate_strength_ok = (
                        signal_smooth[candidate] >= 0.55 * median_peak_signal
                    )

                    too_close_existing = any(
                        min(abs(candidate - q), n_signal - abs(candidate - q))
                        < 0.35 * median_spacing
                        for q in list(consensus_peaks) + added_peaks
                    )

                    local_spacing_ok = (
                        left >= 0.30 * median_spacing and
                        right >= 0.30 * median_spacing
                    )

                    if candidate_strength_ok and local_spacing_ok and not too_close_existing:
                        added_peaks.append(candidate)

        if added_peaks:
            consensus_peaks = np.array(
                sorted(set(list(consensus_peaks) + added_peaks)),
                dtype=int,
            )

            consensus_points = [
                base["ring_points"][int(p)] for p in consensus_peaks
            ]

            support_counts = [
                max(
                    1,
                    len(set(
                        e["ring_id"]
                        for e in peak_entries
                        if min(
                            abs(int(e["peak_index"]) - int(p)),
                            n_signal - abs(int(e["peak_index"]) - int(p))
                        ) <= consensus_window
                    ))
                )
                for p in consensus_peaks
            ]

    # -------------------------
    # Plateau/region fallback for severe undercount cases
    # -------------------------
    if len(consensus_peaks) >= 5:
        signal_smooth = base["signal_smooth"]
        n_signal = len(signal_smooth)

        consensus_peaks = np.array(sorted(consensus_peaks), dtype=int)
        spacings = np.diff(
            np.r_[consensus_peaks, consensus_peaks[0] + n_signal]
        )

        median_spacing_tmp = float(np.median(spacings))
        spacing_cv_tmp = float(
            np.std(spacings) / (np.mean(spacings) + 1e-6)
        )
        max_to_median_tmp = float(
            np.max(spacings) / (median_spacing_tmp + 1e-6)
        )

        severe_undercount_signature = (
            spacing_cv_tmp > 0.32 and
            max_to_median_tmp > 2.2
        )

        if severe_undercount_signature:
            thr = np.percentile(signal_smooth, 65)
            binary = signal_smooth > thr

            binary_pad = np.r_[binary[-20:], binary, binary[:20]]
            binary_pad = ndi.binary_closing(binary_pad, iterations=2)
            binary = binary_pad[20:-20]

            labels, n_regions = ndi.label(binary)

            region_peaks = []

            for lab in range(1, n_regions + 1):
                idx = np.where(labels == lab)[0]

                if len(idx) < 3:
                    continue

                p = int(idx[np.argmax(signal_smooth[idx])])
                region_peaks.append(p)

            region_peaks = np.array(sorted(set(region_peaks)), dtype=int)

            if (
                len(region_peaks) > len(consensus_peaks)
                and len(region_peaks) <= len(consensus_peaks) + 8
            ):
                consensus_peaks = region_peaks
                consensus_points = [
                    base["ring_points"][int(p)] for p in consensus_peaks
                ]
                support_counts = [1 for _ in consensus_peaks]

                base["used_plateau_fallback"] = True

        # -------------------------
        # Exodermis-only spacing rescue
        # Deterministic gap fill for missed exodermal cells.
        # -------------------------
        if enable_spacing_rescue and tissue == "Exodermis" and len(consensus_peaks) >= 5:
            signal_smooth = base["signal_smooth"]
            n_signal = len(signal_smooth)

            consensus_peaks = np.array(sorted(consensus_peaks), dtype=int)
            spacings = np.diff(
                np.r_[consensus_peaks, consensus_peaks[0] + n_signal]
            )

            median_spacing = float(np.median(spacings))
            spacing_cv = float(np.std(spacings) / (np.mean(spacings) + 1e-6))
            max_to_median = float(np.max(spacings) / (median_spacing + 1e-6))

            base["expected_cells_from_spacing"] = (
                int(round(n_signal / median_spacing)) if median_spacing > 0 else 0
            )

            spacing_rescue_added = []

            # Only allow spacing rescue on clearly broken rings
            allow_spacing_rescue = (
                spacing_cv > 0.30 or
                max_to_median > 2.2
            )

            if not allow_spacing_rescue:
                base["spacing_rescue_applied"] = False
                base["spacing_rescue_added"] = 0
                base["spacing_rescue_added_peaks"] = []
            else:

                for i, gap in enumerate(spacings):
                    if median_spacing <= 0:
                        continue

                    # Rescue medium + large missing sectors
                    if gap < 1.35 * median_spacing:
                        continue

                    p1 = int(consensus_peaks[i])
                    p2 = int(consensus_peaks[(i + 1) % len(consensus_peaks)])

                    n_missing = int(round(gap / median_spacing)) - 1
                    n_missing = max(1, min(n_missing, 5))

                    for k in range(1, n_missing + 1):
                        expected = int((p1 + k * gap / (n_missing + 1)) % n_signal)

                        # Refine to strongest nearby signal, but do not let signal dominate.
                        search_radius = max(2, int(0.25 * median_spacing))
                        search_idx = np.arange(
                            expected - search_radius,
                            expected + search_radius + 1
                        ) % n_signal

                        candidate = int(search_idx[np.argmax(signal_smooth[search_idx])])

                        too_close = any(
                            min(abs(candidate - q), n_signal - abs(candidate - q))
                            < 0.35 * median_spacing
                            for q in list(consensus_peaks) + spacing_rescue_added
                        )

                        if not too_close:
                            spacing_rescue_added.append(candidate)

                if spacing_rescue_added:
                    candidate_peaks = np.array(
                        sorted(set(list(consensus_peaks) + spacing_rescue_added)),
                        dtype=int,
                    )

                    expected_cells = (
                        int(round(n_signal / median_spacing)) if median_spacing > 0 else len(candidate_peaks)
                    )

                    severe_ring = (
                        spacing_cv > 0.40 or
                        max_to_median > 3.0
                    )

                    # If rescue inflates a non-severe ring beyond spacing expectation,
                    # reject the rescue. This prevents hallucinated extra cells.
                    if (not severe_ring) and (len(candidate_peaks) > expected_cells + 1):
                        base["spacing_rescue_applied"] = False
                        base["spacing_rescue_added"] = 0
                        base["spacing_rescue_added_peaks"] = []
                        base["spacing_rescue_rejected_overcount"] = True

                    else:
                        consensus_peaks = candidate_peaks

                        consensus_points = [
                            base["ring_points"][int(p)]
                            for p in consensus_peaks
                        ]

                        support_counts = [
                            max(
                                1,
                                len(set(
                                    e["ring_id"]
                                    for e in peak_entries
                                    if min(
                                        abs(int(e["peak_index"]) - int(p)),
                                        n_signal - abs(int(e["peak_index"]) - int(p))
                                    ) <= consensus_window
                                ))
                            )
                            for p in consensus_peaks
                        ]

                        base["spacing_rescue_applied"] = True
                        base["spacing_rescue_added"] = int(len(spacing_rescue_added))
                        base["spacing_rescue_added_peaks"] = spacing_rescue_added
                        base["spacing_rescue_rejected_overcount"] = False
                else:
                    base["spacing_rescue_applied"] = False
                    base["spacing_rescue_added"] = 0
                    base["spacing_rescue_added_peaks"] = []
        

        # -------------------------
        # Exodermis-only radial sanity filter
        # Removes false peaks caused by interrupted exodermis sectors
        # where the sampled point jumps inward/outward.
        # -------------------------
        if tissue == "Exodermis" and len(consensus_peaks) >= 5:
            pts = np.array(consensus_points, dtype=float)

            # Estimate root center from vascular mask if available
            vascular = bio7["Vascular"] > 0
            if vascular.any():
                ys, xs = np.where(vascular)
            else:
                ys, xs = np.where(bio7["Whole Root"] > 0)

            cy = float(np.mean(ys))
            cx = float(np.mean(xs))

            radii = np.sqrt((pts[:, 0] - cy) ** 2 + (pts[:, 1] - cx) ** 2)

            median_radius = float(np.median(radii))

            # Robust allowed radial band
            lower_radius = 0.75 * median_radius
            upper_radius = 1.25 * median_radius

            keep = (
                (radii >= lower_radius) &
                (radii <= upper_radius)
            )

            if np.sum(~keep) > 0:
                base["radial_sanity_applied"] = True
                base["radial_sanity_removed"] = int(np.sum(~keep))
                base["radial_sanity_removed_peaks"] = [
                    int(p) for p in np.array(consensus_peaks)[~keep]
                ]
            else:
                base["radial_sanity_applied"] = False
                base["radial_sanity_removed"] = 0
                base["radial_sanity_removed_peaks"] = []

            consensus_peaks = np.array(consensus_peaks)[keep].astype(int)
            consensus_points = [pt for pt, k in zip(consensus_points, keep) if k]
            support_counts = [s for s, k in zip(support_counts, keep) if k]

    signal_smooth = base["signal_smooth"]

    # -------------------------
    # Final spacing sanity collapse on multi-ring consensus
    # -------------------------
    peaks, spacing_sanity_info = spacing_sanity_collapse_peaks(
        signal_smooth=signal_smooth,
        peaks=consensus_peaks,
        min_spacing_ratio_trigger=preset_params["min_spacing_ratio_trigger"],
        spacing_cv_trigger=preset_params["spacing_cv_trigger"],
        weak_peak_trigger=preset_params["weak_peak_trigger"],
        severity_threshold=preset_params["severity_threshold"],
        max_iter=5,
    )

    consensus_peaks = np.array(sorted(peaks), dtype=int)

    consensus_points = [
        base["ring_points"][int(p)]
        for p in consensus_peaks
    ]

    # -------------------------
    # Final radial sanity filter
    # Removes false peaks caused by lateral roots / interrupted tissue sectors.
    # These often map to the root center because no valid exodermal ring point exists.
    # -------------------------
    if tissue in ["Exodermis", "Endodermis"] and len(consensus_peaks) >= 5:
        pts = np.array(consensus_points, dtype=float)

        vascular = bio7["Vascular"] > 0
        if vascular.any():
            ys, xs = np.where(vascular)
        else:
            ys, xs = np.where(bio7["Whole Root"] > 0)

        cy = float(np.mean(ys))
        cx = float(np.mean(xs))

        radii = np.sqrt((pts[:, 0] - cy) ** 2 + (pts[:, 1] - cx) ** 2)

        # Use only non-central points to define expected ring radius
        valid_radii = radii[radii > np.percentile(radii, 20)]

        if len(valid_radii) >= 5:
            median_radius = float(np.median(valid_radii))

            lower_radius = 0.70 * median_radius
            upper_radius = 1.30 * median_radius

            keep = (
                (radii >= lower_radius) &
                (radii <= upper_radius)
            )

            if np.sum(~keep) > 0:
                base["radial_sanity_applied"] = True
                base["radial_sanity_removed"] = int(np.sum(~keep))
                base["radial_sanity_removed_peaks"] = [
                    int(p) for p in consensus_peaks[~keep]
                ]

                consensus_peaks = consensus_peaks[keep]
                consensus_points = [pt for pt, k in zip(consensus_points, keep) if k]

            else:
                base["radial_sanity_applied"] = False
                base["radial_sanity_removed"] = 0
                base["radial_sanity_removed_peaks"] = []

    # ── DAPI cross-channel validation ─────────────────────────────────────────
    # Validates consensus peaks against the DAPI (calcofluor white) channel.
    # Real cell walls are bright in DAPI at the peak angular position.
    # EFI-projected Casparian strip signal lands in cell interiors (not walls)
    # and therefore has weak DAPI support → those peaks are rejected.
    # Only active when dapi_validation=True and the primary channel is not DAPI.
    if dapi_validation and channel != "DAPI" and len(consensus_peaks) > 0:
        _dapi_img = raw[..., 2].astype(float)    # DAPI is always index 2

        # DAPI intensity at each consensus peak's spatial position
        _dapi_at_peaks = np.array(
            [_dapi_img[int(pt[0]), int(pt[1])] for pt in consensus_points],
            dtype=float,
        )

        # Baseline: 20th-percentile DAPI across the full ring
        _base_pts = base["ring_points"]
        _dapi_ring = np.array(
            [_dapi_img[int(ry), int(rx)] for ry, rx in _base_pts],
            dtype=float,
        )
        _dapi_baseline = float(np.percentile(_dapi_ring, 20))

        _dapi_adj  = np.maximum(_dapi_at_peaks - _dapi_baseline, 0.0)
        _dapi_median = float(np.median(_dapi_adj))

        if _dapi_median > 0:
            _valid = _dapi_adj >= dapi_threshold_fraction * _dapi_median
            consensus_peaks  = consensus_peaks[_valid]
            consensus_points = [p for p, v in zip(consensus_points, _valid) if v]
            support_counts   = [s for s, v in zip(support_counts,   _valid) if v]
            base["dapi_validation_applied"]  = True
            base["dapi_validation_rejected"] = int(np.sum(~_valid))
        else:
            base["dapi_validation_applied"]  = False
            base["dapi_validation_rejected"] = 0
    else:
        base["dapi_validation_applied"]  = False
        base["dapi_validation_rejected"] = 0

    final_count = int(len(consensus_peaks))

    # -------------------------
    # Exodermis-only valid-arc spacing correction
    # Estimates expected count only across valid exodermis arc,
    # not the full 360° ring.
    # -------------------------
    if tissue == "Exodermis" and len(consensus_peaks) >= 5:
        signal_smooth = base["signal_smooth"]
        signal_len = len(signal_smooth)

        spacings = np.diff(
            np.r_[consensus_peaks, consensus_peaks[0] + signal_len]
        )

        median_spacing = float(np.median(spacings))
        spacing_cv_corr = float(np.std(spacings) / (np.mean(spacings) + 1e-6))
        max_to_median_corr = float(np.max(spacings) / (median_spacing + 1e-6))

        # Estimate valid exodermis arc from ring point radius + signal support
        vascular = bio7["Vascular"] > 0
        if vascular.any():
            ys, xs = np.where(vascular)
        else:
            ys, xs = np.where(bio7["Whole Root"] > 0)

        cy = float(np.mean(ys))
        cx = float(np.mean(xs))

        all_ring_pts = np.array(base["ring_points"], dtype=float)
        ring_radii = np.sqrt(
            (all_ring_pts[:, 0] - cy) ** 2 +
            (all_ring_pts[:, 1] - cx) ** 2
        )

        valid_radius_ref = ring_radii[ring_radii > np.percentile(ring_radii, 20)]

        if len(valid_radius_ref) >= 5:
            median_radius = float(np.median(valid_radius_ref))
        else:
            median_radius = float(np.median(ring_radii))

        radius_valid = (
            (ring_radii >= 0.70 * median_radius) &
            (ring_radii <= 1.30 * median_radius)
        )

        signal_valid = signal_smooth >= np.percentile(signal_smooth, 20)

        valid_arc = radius_valid & signal_valid
        valid_arc = ndi.binary_closing(valid_arc, iterations=3)
        valid_arc = ndi.binary_opening(valid_arc, iterations=2)

        valid_arc_len = int(np.sum(valid_arc))
        valid_arc_fraction = float(valid_arc_len / signal_len)

        if median_spacing > 0 and valid_arc_len > 0:
            spacing_count_arc = int(round(valid_arc_len / median_spacing))
            spacing_count_full = int(round(signal_len / median_spacing))

            # Do not allow arc estimate to shrink too aggressively
            spacing_count = max(spacing_count_arc, int(0.75 * spacing_count_full))
        else:
            spacing_count = final_count

        base["spacing_count"] = spacing_count
        base["valid_arc_len"] = valid_arc_len
        base["valid_arc_fraction"] = valid_arc_fraction
        base["spacing_correction_spacing_cv"] = spacing_cv_corr
        base["spacing_correction_max_to_median"] = max_to_median_corr

        if spacing_cv_corr > 0.35 or max_to_median_corr > 2.0:
            base["spacing_correction_applied"] = True

            # If valid-arc estimate became too conservative, fall back to full-ring estimate
            if spacing_count <= final_count and median_spacing > 0:
                spacing_count = int(round(signal_len / median_spacing))

            # Never reduce count with this correction
            spacing_count = max(spacing_count, final_count)

            base["spacing_correction_added"] = int(spacing_count - final_count)
            final_count = spacing_count
        else:
            base["spacing_correction_applied"] = False
            base["spacing_correction_added"] = 0

    else:
        base["spacing_count"] = final_count
        base["valid_arc_len"] = ""
        base["valid_arc_fraction"] = ""
        base["spacing_correction_spacing_cv"] = ""
        base["spacing_correction_max_to_median"] = ""
        base["spacing_correction_applied"] = False
        base["spacing_correction_added"] = 0

    base["count"] = final_count
    base["peaks"] = consensus_peaks
    base["peak_points"] = consensus_points
    base["multi_ring_positions"] = positions
    base["multi_ring_support"] = support_counts
    base["multi_ring_all_stats"] = all_stats

    base["raw_count_before_spacing_sanity"] = int(spacing_sanity_info["raw_count"])
    base["spacing_sanity_applied"] = bool(spacing_sanity_info["applied"])
    base["spacing_sanity_removed"] = int(spacing_sanity_info["removed"])
    base["spacing_sanity_removed_peaks"] = spacing_sanity_info.get("removed_peaks", [])
    base["spacing_sanity_removed_values"] = spacing_sanity_info.get("removed_peak_values", [])
    base["spacing_sanity_removed_spacings"] = spacing_sanity_info.get("removed_peak_spacings", [])
    base["spacing_sanity_reason"] = spacing_sanity_info["reason"]
    base["count_status"] = (
        "spacing_sanity_corrected"
        if spacing_sanity_info["applied"]
        else base.get("count_status", "")
    )
    base["correction_applied"] = -int(spacing_sanity_info["removed"])

    base["likely_missed_cell"] = any(
        s.get("likely_missed_cell", False) for s in all_stats
    )

    base["has_unresolved_wide_peak"] = any(
        s.get("has_unresolved_wide_peak", False) for s in all_stats
    )

    base["wide_peak_fraction"] = max(
        [s.get("wide_peak_fraction", 0) for s in all_stats] or [0]
    )

    base["median_peak_width"] = np.median([
        s.get("median_peak_width", 0) for s in all_stats
    ])

    # redo basic QC on consensus peaks
    if len(consensus_peaks) >= 3:
        signal_smooth = base["signal_smooth"]
        spacings = np.diff(np.r_[consensus_peaks, consensus_peaks[0] + len(signal_smooth)])

        base["mean_spacing"] = float(np.mean(spacings))
        base["spacing_cv"] = float(np.std(spacings) / (np.mean(spacings) + 1e-6))
        base["median_spacing"] = float(np.median(spacings))
        base["min_spacing"] = float(np.min(spacings))
        base["max_spacing"] = float(np.max(spacings))
        base["min_to_median_spacing"] = float(np.min(spacings) / (np.median(spacings) + 1e-6))
        base["max_to_median_spacing"] = float(np.max(spacings) / (np.median(spacings) + 1e-6))

        peak_values = signal_smooth[consensus_peaks]
        base["median_peak_signal"] = float(np.median(peak_values))
        base["min_peak_signal"] = float(np.min(peak_values))
        base["weak_peak_fraction"] = float(np.mean(peak_values < 0.5 * np.median(peak_values)))
        base["peak_cv"] = float(np.std(peak_values) / (np.mean(peak_values) + 1e-6))

    else:
        base["confidence"] = "low"
        base["review_needed"] = True

    # -------------------------
    # QC-based error classification
    # -------------------------
    spacing_cv = base.get("spacing_cv", 999)
    max_gap = base.get("max_to_median_spacing", 999)
    min_gap = base.get("min_to_median_spacing", 0)
    weak_frac = base.get("weak_peak_fraction", 1)
    peak_cv = base.get("peak_cv", 999)

    qc_reasons = []

    if spacing_cv > 0.30:
        qc_reasons.append("high_spacing_cv")

    if max_gap > 2.20:
        qc_reasons.append("large_gap")

    if min_gap < 0.48:
        qc_reasons.append("small_gap")

    if weak_frac > 0.25:
        qc_reasons.append("many_weak_peaks")

    if peak_cv > 0.40:
        qc_reasons.append("high_peak_cv")

    if base.get("has_unresolved_wide_peak", False):
        qc_reasons.append("unresolved_wide_peak")

    if base.get("radial_sanity_removed", 0) > 0:
        qc_reasons.append("radial_issue")

    if base.get("spacing_sanity_applied", False):
        qc_reasons.append("spacing_sanity_corrected")

    if base.get("spacing_rescue_applied", False):
        qc_reasons.append("spacing_rescue_applied")

    if base.get("spacing_correction_applied", False):
        qc_reasons.append("spacing_correction_applied")

    if base.get("used_plateau_fallback", False):
        qc_reasons.append("plateau_fallback")

    if base.get("rescue_skipped_unstable_ring", False):
        qc_reasons.append("rescue_skipped_unstable_ring")

    if base.get("rescue_skipped_compressed_ring", False):
        qc_reasons.append("rescue_skipped_compressed_ring")

    if base.get("spacing_rescue_rejected_overcount", False):
        qc_reasons.append("spacing_rescue_rejected_overcount")

    base["qc_reason"] = ";".join(qc_reasons)
    base["review_needed"] = len(qc_reasons) > 0
    base["needs_manual_review"] = base["review_needed"]

    base["qc_clean"] = not base["review_needed"]
    base["likely_undercount"] = (
        max_gap > 1.8 or
        (spacing_cv > 0.35 and max_gap > 1.5)
    )
    base["likely_overcount"] = (
        weak_frac > 0.25 and
        min_gap < 0.55
    )

    if base["qc_clean"]:
        base["count_status"] = "clean"
    elif base["likely_undercount"] and not base["likely_overcount"]:
        base["count_status"] = "likely_undercount"
    elif base["likely_overcount"] and not base["likely_undercount"]:
        base["count_status"] = "likely_overcount"
    else:
        base["count_status"] = "review"

    base["raw_count_before_spacing_sanity"] = int(spacing_sanity_info["raw_count"])
    base["spacing_sanity_applied"] = bool(spacing_sanity_info["applied"])
    base["spacing_sanity_removed"] = int(spacing_sanity_info["removed"])
    base["spacing_sanity_reason"] = spacing_sanity_info["reason"]
    base["count_status"] = (
        "spacing_sanity_corrected"
        if spacing_sanity_info["applied"]
        else base.get("count_status", "")
    )
    base["correction_applied"] = -int(spacing_sanity_info["removed"])
    base["spacing_sanity_removed_peaks"] = spacing_sanity_info.get("removed_peaks", [])
    base["spacing_sanity_removed_values"] = spacing_sanity_info.get("removed_peak_values", [])
    base["spacing_sanity_removed_spacings"] = spacing_sanity_info.get("removed_peak_spacings", [])

    return base


def suggest_ring_range(
    raw,
    bio7,
    tissue,
    channel,
    ring_width,
    n_points,
    smooth_sigma,
    peak_distance,
    peak_prominence,
    local_contrast_threshold,
    peak_width,
    merge_distance,
    ring_consensus_min,
    ring_position_iterations,
    enable_gap_fill=False,
    enable_dense_check=False,
    enable_wide_peak_split=False,
    enable_close_doublet_collapse=True,
    auto_peak_distance=False,
    enable_adaptive_peak_cleanup=False,
):
    rows = []

    for label, (rmin, rmax) in RING_RANGE_PRESETS.items():
        stats = multi_ring_peak_count(
            raw=raw,
            bio7=bio7,
            tissue=tissue,
            channel=channel,
            ring_position_min=rmin,
            ring_position_max=rmax,
            ring_position_iterations=ring_position_iterations,
            ring_consensus_min=ring_consensus_min,
            ring_width=ring_width,
            n_points=n_points,
            smooth_sigma=smooth_sigma,
            peak_distance=peak_distance,
            peak_prominence=peak_prominence,
            local_contrast_threshold=local_contrast_threshold,
            peak_width=peak_width,
            merge_distance=merge_distance,
            enable_gap_fill=enable_gap_fill,
            enable_dense_check=enable_dense_check,
            enable_wide_peak_split=enable_wide_peak_split,
            enable_close_doublet_collapse=enable_close_doublet_collapse,
            auto_peak_distance=auto_peak_distance,
            enable_adaptive_peak_cleanup=enable_adaptive_peak_cleanup,
            enable_spacing_rescue=enable_spacing_rescue,
            enable_valley_rescue=enable_valley_rescue,
        )

        spacing_cv = float(stats.get("spacing_cv", 999))
        min_ratio = float(stats.get("min_to_median_spacing", 0))
        max_ratio = float(stats.get("max_to_median_spacing", 999))
        weak_frac = float(stats.get("weak_peak_fraction", 1))
        peak_cv = float(stats.get("peak_cv", 999))

        score = (
            3.0 * spacing_cv
            + 2.0 * max(0, 0.50 - min_ratio)
            + 1.5 * max(0, max_ratio - 1.50)
            + 2.0 * weak_frac
            + 1.0 * peak_cv
        )

        rows.append({
            "ring_range_label": label,
            "ring_position_min": rmin,
            "ring_position_max": rmax,
            "score": score,
            "count": stats.get("count", np.nan),
            "spacing_cv": spacing_cv,
            "min_to_median_spacing": min_ratio,
            "max_to_median_spacing": max_ratio,
            "weak_peak_fraction": weak_frac,
            "peak_cv": peak_cv,
            "review_needed": stats.get("review_needed", True),
            "confidence": stats.get("confidence", "low"),
        })

    df = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    return df


def dual_channel_ring_peak_count(
    raw,
    bio7,
    tissue,
    primary_channel="TRITC",
    secondary_channel="FITC",
    **kwargs,
):
    """
    Run ring counting on two channels and choose the cleaner channel only
    when QC strongly supports switching. Otherwise keep the primary channel.
    """

    def channel_qc_score(stats):
        spacing_cv = float(stats.get("spacing_cv", 999))
        peak_cv = float(stats.get("peak_cv", 999))
        weak_frac = float(stats.get("weak_peak_fraction", 1))
        min_gap = float(stats.get("min_to_median_spacing", 0))
        max_gap = float(stats.get("max_to_median_spacing", 999))

        # Lower score = cleaner channel.
        return (
            1.5 * spacing_cv
            + 1.0 * peak_cv
            + 2.0 * weak_frac
            + 1.0 * max(0, 0.50 - min_gap)
            + 0.75 * max(0, max_gap - 1.60)
        )

    stats_primary = multi_ring_peak_count(
        raw=raw,
        bio7=bio7,
        tissue=tissue,
        channel=primary_channel,
        **kwargs,
    )

    stats_secondary = multi_ring_peak_count(
        raw=raw,
        bio7=bio7,
        tissue=tissue,
        channel=secondary_channel,
        **kwargs,
    )

    primary_score = channel_qc_score(stats_primary)
    secondary_score = channel_qc_score(stats_secondary)

    primary_count = int(stats_primary["count"])
    secondary_count = int(stats_secondary["count"])
    channel_difference = secondary_count - primary_count

    primary_review = bool(stats_primary.get("review_needed", False))
    secondary_review = bool(stats_secondary.get("review_needed", False))

    # Conservative switching logic:
    # 1. Switch if primary is flagged and secondary is clean.
    # 2. Or switch if counts are close and secondary is clearly cleaner.
    # 3. Otherwise keep primary.
    use_secondary = False
    selection_reason = "primary_default"

    if primary_review and not secondary_review:
        use_secondary = True
        selection_reason = "primary_flagged_secondary_clean"

    elif (
        abs(channel_difference) <= 2
        and secondary_score < 0.95 * primary_score
    ):
        use_secondary = True
        selection_reason = "secondary_better_qc_close_count"

    elif (
        abs(channel_difference) > 2
        and secondary_score < 0.80 * primary_score
    ):
        use_secondary = True
        selection_reason = "secondary_much_better_qc_large_diff"

    elif (
        abs(channel_difference) >= 3
        and secondary_score < primary_score
    ):
        use_secondary = True
        selection_reason = "secondary_better_qc_large_diff"

    if use_secondary:
        out = stats_secondary.copy()
        selected_channel = secondary_channel
    else:
        out = stats_primary.copy()
        selected_channel = primary_channel

    out["channel"] = selected_channel
    out["channel_used_for_count"] = selected_channel
    out["channel_selection_reason"] = selection_reason

    out["primary_channel"] = primary_channel
    out["secondary_channel"] = secondary_channel

    out["primary_channel_count"] = primary_count
    out["secondary_channel_count"] = secondary_count
    out["channel_difference"] = channel_difference
    out["channel_disagreement"] = abs(channel_difference) > 1

    out["primary_channel_qc_score"] = float(primary_score)
    out["secondary_channel_qc_score"] = float(secondary_score)
    out["primary_channel_review_needed"] = primary_review
    out["secondary_channel_review_needed"] = secondary_review
    out["primary_channel_qc_reason"] = stats_primary.get("qc_reason", "")
    out["secondary_channel_qc_reason"] = stats_secondary.get("qc_reason", "")

    # If both channels disagree strongly, keep review flag even after selection.
    out["review_needed"] = bool(
        out.get("review_needed", False)
        or abs(channel_difference) > 2
    )

    if out["review_needed"]:
        out["confidence"] = "low"
    elif abs(channel_difference) == 1:
        out["confidence"] = "medium"
    else:
        out["confidence"] = out.get("confidence", "high")

    print(
        f"DUAL QC | primary={primary_channel}:{primary_count} "
        f"secondary={secondary_channel}:{secondary_count} "
        f"selected={selected_channel} "
        f"reason={selection_reason} "
        f"diff={channel_difference}"
    )

    return out
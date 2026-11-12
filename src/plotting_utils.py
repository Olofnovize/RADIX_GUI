import numpy as np
import cv2
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CHANNELS = ["TRITC", "FITC", "DAPI"]

def to_uint8(img):
    img = img - img.min()
    img = img / (img.max() + 1e-6)
    return (img * 255).astype(np.uint8)


def crop_to_mask(img, mask, pad=80):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return img, (0, img.shape[0], 0, img.shape[1])

    y0 = max(0, ys.min() - pad)
    y1 = min(mask.shape[0], ys.max() + pad)
    x0 = max(0, xs.min() - pad)
    x1 = min(mask.shape[1], xs.max() + pad)

    return img[y0:y1, x0:x1], (y0, y1, x0, x1)


def make_ring_qc_plot(base_img, ring_stats, crop_offset=(0, 0), selected_peak_id=None, channel=None, bad_peak_ids=None):
    """
    Interactive QC plot:
    left = image with ring path and peak nodes
    right = ring signal with detected peaks and prominence guide

    Note: prominence is not a strict absolute intensity threshold;
    it is a local peak-prominence criterion. The dashed line is a visual guide.
    """

    yoff, xoff = crop_offset
    h, w = base_img.shape[:2]

    signal = ring_stats["signal_smooth"]
    peaks = ring_stats["peaks"]
    peak_values = signal[peaks]

    peak_ids = np.arange(1, len(peaks) + 1)

    channel_label = channel or ring_stats.get("channel", "selected channel")

    ring_xy = []
    for y, x in ring_stats["ring_points"]:
        yy = y - yoff
        xx = x - xoff
        if 0 <= yy < h and 0 <= xx < w:
            ring_xy.append((xx, yy))

    peak_xy = []
    kept_peak_ids = []
    kept_peak_indices = []
    kept_peak_values = []

    for pid, p, val, (y, x) in zip(
        peak_ids,
        peaks,
        peak_values,
        ring_stats["peak_points"],
    ):
        yy = y - yoff
        xx = x - xoff
        if 0 <= yy < h and 0 <= xx < w:
            peak_xy.append((xx, yy))
            kept_peak_ids.append(pid)
            kept_peak_indices.append(p)
            kept_peak_values.append(val)

    ring_x = [p[0] for p in ring_xy]
    ring_y = [p[1] for p in ring_xy]

    peak_x = [p[0] for p in peak_xy]
    peak_y = [p[1] for p in peak_xy]

    _bad = set(bad_peak_ids) if bad_peak_ids else set()

    marker_sizes = []
    marker_colors = []

    for pid in kept_peak_ids:
        if selected_peak_id is not None and pid == selected_peak_id:
            marker_sizes.append(18)
            marker_colors.append("yellow")
        elif pid in _bad:
            marker_sizes.append(14)
            marker_colors.append("orange")
        else:
            marker_sizes.append(10)
            marker_colors.append("#00cc44")

    signal_marker_sizes = []
    signal_marker_colors = []

    for pid in peak_ids:
        if selected_peak_id is not None and pid == selected_peak_id:
            signal_marker_sizes.append(14)
            signal_marker_colors.append("yellow")
        elif pid in _bad:
            signal_marker_sizes.append(12)
            signal_marker_colors.append("orange")
        else:
            signal_marker_sizes.append(8)
            signal_marker_colors.append("#00cc44")

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.52, 0.48],
        subplot_titles=("Ring path + detected peaks", f"Ring signal ({channel_label})"),
    )

    # left: image
    fig.add_trace(
        go.Image(z=base_img),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ring_x,
            y=ring_y,
            mode="markers",
            marker=dict(size=2, color="rgba(160,160,160,0.5)"),
            hoverinfo="skip",
            name="Ring path",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=peak_x,
            y=peak_y,
            mode="markers+text",
            text=[str(pid) for pid in kept_peak_ids],
            textposition="top center",
            marker=dict(
                size=marker_sizes,
                color=marker_colors,
                line=dict(color="white", width=1),
            ),
            customdata=np.column_stack([
                kept_peak_ids,
                kept_peak_indices,
                np.round(kept_peak_values, 1),
            ]),
            hovertemplate=(
                "Peak ID: %{customdata[0]}<br>"
                "Signal index: %{customdata[1]}<br>"
                "Signal value: %{customdata[2]}<br>"
                "x=%{x}, y=%{y}<extra></extra>"
            ),
            name="Detected peaks",
        ),
        row=1,
        col=1,
    )

    fig.update_xaxes(visible=False, range=[0, w], row=1, col=1)
    fig.update_yaxes(visible=False, range=[h, 0], row=1, col=1)

    # right: signal
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(signal)),
            y=signal,
            mode="lines",
            name="Smoothed ring signal",
            line=dict(color="black"),
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=peaks,
            y=peak_values,
            mode="markers+text",
            text=[str(pid) for pid in peak_ids],
            textposition="top center",
            marker=dict(
                size=signal_marker_sizes,
                color=signal_marker_colors,
                line=dict(color="black", width=1),
            ),
            customdata=np.column_stack([
                peak_ids,
                peaks,
                np.round(peak_values, 1),
            ]),
            hovertemplate=(
                "Peak ID: %{customdata[0]}<br>"
                "Signal index: %{customdata[1]}<br>"
                "Signal value: %{customdata[2]}<extra></extra>"
            ),
            name="Detected peaks",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Ring index", row=1, col=2)
    plot_channel = ring_stats.get("channel", "selected channel")
    fig.update_yaxes(
        title_text=f"Background-corrected {plot_channel} signal",
        row=1,
        col=2,
    )

    fig.update_layout(
        height=620,
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
    )

    return fig


def build_peak_feature_table(raw, ring_stats, channel):
    """
    Build a diagnostic feature table for currently accepted ring peaks.
    This does NOT change the count; it only exposes features for rule design.
    """

    if ring_stats is None:
        return pd.DataFrame()

    signal = np.asarray(ring_stats.get("signal_smooth", []), dtype=float)
    peaks = np.asarray(ring_stats.get("peaks", []), dtype=int)

    if signal.size == 0 or peaks.size == 0:
        return pd.DataFrame()

    n_signal = len(signal)

    # Keep everything sorted by ring index
    order = np.argsort(peaks)
    peaks = peaks[order]

    peak_points_all = ring_stats.get("peak_points", [])
    peak_points = [peak_points_all[i] for i in order] if len(peak_points_all) == len(order) else []

    peak_heights = signal[peaks]

    # Prominence and width from the final smoothed signal
    # Manual local contrast/width features.
    # This is more reliable for multi-ring consensus peaks, because final peaks
    # may not be perfect local maxima in the displayed middle-ring signal.
    manual_prominences = []
    manual_widths = []
    manual_left_valleys = []
    manual_right_valleys = []

    local_window = 12

    for p in peaks:
        p = int(p)

        idx_left = np.arange(p - local_window, p) % n_signal
        idx_right = np.arange(p + 1, p + local_window + 1) % n_signal

        peak_val = float(signal[p])
        left_valley = float(np.min(signal[idx_left])) if len(idx_left) else peak_val
        right_valley = float(np.min(signal[idx_right])) if len(idx_right) else peak_val

        valley = max(left_valley, right_valley)
        prom = peak_val - valley

        manual_prominences.append(prom)
        manual_left_valleys.append(left_valley)
        manual_right_valleys.append(right_valley)

        # Approximate local support width above half-prominence height.
        half_height = valley + 0.5 * max(prom, 0)

        width_idx = np.arange(p - local_window, p + local_window + 1) % n_signal
        local_vals = signal[width_idx]
        width = int(np.sum(local_vals >= half_height)) if prom > 0 else 0

        manual_widths.append(width)

    prominences = np.asarray(manual_prominences, dtype=float)
    widths = np.asarray(manual_widths, dtype=float)
    left_bases = np.asarray(manual_left_valleys, dtype=float)
    right_bases = np.asarray(manual_right_valleys, dtype=float)

    # Circular angular spacing
    prev_peaks = np.roll(peaks, 1)
    next_peaks = np.roll(peaks, -1)

    left_spacing = (peaks - prev_peaks) % n_signal
    right_spacing = (next_peaks - peaks) % n_signal

    spacings_all = np.diff(np.r_[peaks, peaks[0] + n_signal])
    median_spacing = float(np.median(spacings_all)) if len(spacings_all) else np.nan

    min_spacing = np.minimum(left_spacing, right_spacing)
    max_spacing = np.maximum(left_spacing, right_spacing)

    min_spacing_ratio = min_spacing / (median_spacing + 1e-6)
    max_spacing_ratio = max_spacing / (median_spacing + 1e-6)

    # XY distances between neighboring peak points
    if len(peak_points) == len(peaks):
        pts = np.asarray(peak_points, dtype=float)  # y, x
        prev_pts = np.roll(pts, 1, axis=0)
        next_pts = np.roll(pts, -1, axis=0)

        xy_dist_left = np.linalg.norm(pts - prev_pts, axis=1)
        xy_dist_right = np.linalg.norm(pts - next_pts, axis=1)
        min_xy_dist = np.minimum(xy_dist_left, xy_dist_right)
        median_xy_dist = float(np.median(min_xy_dist)) if len(min_xy_dist) else np.nan
    else:
        pts = np.full((len(peaks), 2), np.nan)
        xy_dist_left = np.full(len(peaks), np.nan)
        xy_dist_right = np.full(len(peaks), np.nan)
        min_xy_dist = np.full(len(peaks), np.nan)
        median_xy_dist = np.nan

    # Gradient / edge strength in the selected raw channel
    ch_idx = {"TRITC": 0, "FITC": 1, "DAPI": 2}
    img = raw[..., ch_idx[channel]].astype(np.float32)

    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    grad_ref_vals = grad_mag[grad_mag > 0]
    grad_ref = np.percentile(grad_ref_vals, 99) if grad_ref_vals.size else 1.0

    gradient_max_r2 = []
    selected_channel_raw_r2 = []

    # Multi-channel local support around each peak
    multi_channel_raw = {ch: [] for ch in CHANNELS}
    multi_channel_norm = {ch: [] for ch in CHANNELS}

    channel_refs = {}
    for ch in CHANNELS:
        ch_img = raw[..., ch_idx[ch]].astype(np.float32)
        vals = ch_img[ch_img > 0]
        channel_refs[ch] = np.percentile(vals, 99) if vals.size else 1.0

    h, w = img.shape

    for yx in pts:
        y, x = yx

        if np.isnan(y) or np.isnan(x):
            gradient_max_r2.append(np.nan)
            selected_channel_raw_r2.append(np.nan)
            for ch in CHANNELS:
                multi_channel_raw[ch].append(np.nan)
                multi_channel_norm[ch].append(np.nan)
            continue

        y = int(np.clip(round(y), 0, h - 1))
        x = int(np.clip(round(x), 0, w - 1))

        y0 = max(0, y - 2)
        y1 = min(h, y + 3)
        x0 = max(0, x - 2)
        x1 = min(w, x + 3)

        gradient_max_r2.append(float(np.max(grad_mag[y0:y1, x0:x1])))
        selected_channel_raw_r2.append(float(np.max(img[y0:y1, x0:x1])))

        for ch in CHANNELS:
            ch_img = raw[..., ch_idx[ch]].astype(np.float32)
            local_val = float(np.max(ch_img[y0:y1, x0:x1]))
            multi_channel_raw[ch].append(local_val)
            multi_channel_norm[ch].append(local_val / (channel_refs[ch] + 1e-6))

    gradient_max_r2 = np.asarray(gradient_max_r2, dtype=float)
    gradient_norm = gradient_max_r2 / (grad_ref + 1e-6)

    # Multi-ring support if available
    support = ring_stats.get("multi_ring_support", None)
    if support is not None and len(support) == len(peaks):
        support = np.asarray(support)[order]
    else:
        support = np.full(len(peaks), np.nan)

    # Relative features
    median_height = np.nanmedian(peak_heights) if len(peak_heights) else np.nan
    median_prom = np.nanmedian(prominences) if len(prominences) else np.nan
    median_width = np.nanmedian(widths) if len(widths) else np.nan

    height_ratio = peak_heights / (median_height + 1e-6)
    prominence_ratio = prominences / (median_prom + 1e-6)
    width_ratio = widths / (median_width + 1e-6)

    # Rule hints only; not filtering yet
    rule_hint = []
    preliminary_score = []

    for i in range(len(peaks)):
        score = 0
        reasons = []

        if height_ratio[i] >= 0.40:
            score += 1
        else:
            reasons.append("weak height")

        if prominence_ratio[i] >= 0.35:
            score += 1
        else:
            reasons.append("weak prominence")

        if 0.35 <= width_ratio[i] <= 3.0:
            score += 1
        else:
            reasons.append("implausible width")

        if min_spacing_ratio[i] >= 0.45:
            score += 1
        else:
            reasons.append("too close angularly")

        if not np.isnan(min_xy_dist[i]) and not np.isnan(median_xy_dist):
            if min_xy_dist[i] >= 0.25 * median_xy_dist:
                score += 1
            else:
                reasons.append("too close spatially")

        if not np.isnan(support[i]):
            if support[i] >= 3:
                score += 1
            else:
                reasons.append("low multi-ring support")

        if gradient_norm[i] >= 0.10:
            score += 1
        else:
            reasons.append("weak gradient")

        preliminary_score.append(score)
        rule_hint.append("plausible" if len(reasons) == 0 else "; ".join(reasons))

    df = pd.DataFrame({
        "peak_id": np.arange(1, len(peaks) + 1),
        "ring_index": peaks,
        "y": pts[:, 0] if len(pts) else np.nan,
        "x": pts[:, 1] if len(pts) else np.nan,

        "signal_height": peak_heights,
        "height_ratio": height_ratio,

        "prominence": prominences,
        "prominence_ratio": prominence_ratio,

        "left_valley": left_bases,
        "right_valley": right_bases,

        "width": widths,
        "width_ratio": width_ratio,

        "left_spacing": left_spacing,
        "right_spacing": right_spacing,
        "min_spacing_ratio": min_spacing_ratio,
        "max_spacing_ratio": max_spacing_ratio,

        "xy_dist_left": xy_dist_left,
        "xy_dist_right": xy_dist_right,
        "min_xy_dist": min_xy_dist,

        "multi_ring_support": support,

        "gradient_max_r2": gradient_max_r2,
        "gradient_norm": gradient_norm,

        "selected_channel_raw_r2": selected_channel_raw_r2,

        "TRITC_norm_r2": multi_channel_norm["TRITC"],
        "FITC_norm_r2": multi_channel_norm["FITC"],
        "DAPI_norm_r2": multi_channel_norm["DAPI"],

        "preliminary_score": preliminary_score,
        "rule_hint": rule_hint,
    })

    return df
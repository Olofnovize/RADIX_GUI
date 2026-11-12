import subprocess
import sys
import re

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path

from src.ui_widgets import range_slider_with_boxes

from src.io_utils import (
    load_sample,
    infer_species,
)

from src.intensity_quant import (
    BARRIER_TISSUES,
    normalize_channel,
    normalize_percentile_rgb,
    compute_threshold_percentile_suggestions,
    measure_barrier_intensity_batch,
    measure_root_morphometrics,
)

from src.plotting_utils import (
    to_uint8,
    crop_to_mask,
)


# -------------------------
# SETTINGS
# -------------------------

st.set_page_config(layout="wide")

st.sidebar.markdown("## Input folders")

data_dir_input = st.sidebar.text_input(
    "Image folder",
    "data",
    key="data_dir",
    help=(
        "Folder containing one subfolder per sample, each with its "
        "TRITC/FITC/DAPI TIFF files. Defaults to the 'data' folder next "
        "to this app — copy your images there, or type a full path."
    ),
)

model_dir_input = st.sidebar.text_input(
    "Model checkpoint file or folder",
    "model",
    key="model_dir",
    help=(
        "Defaults to the bundled 'model' folder next to this app "
        "(contains the trained RADIX checkpoint). Only change this if "
        "you were given a different checkpoint."
    ),
)

DATA_DIR = Path(data_dir_input)
EXP_DIR = DATA_DIR.parent
ML_DIR = EXP_DIR / "ml"
PRED_DIR = ML_DIR / "predictions"
MODEL_DIR = Path(model_dir_input)

pred_dir_override = st.sidebar.text_input(
    "Prediction folder (leave blank for auto)",
    "",
    key="pred_dir_override",
    help=(
        "By default predictions are saved to <image_folder>/../ml/predictions/. "
        "Set this only if your predictions are stored somewhere else."
    ),
)

if pred_dir_override.strip():
    PRED_DIR = Path(pred_dir_override.strip())

# Always use prediction masks in the lab version
mask_source = "Prediction (.npy)"

if not data_dir_input.strip() or not model_dir_input.strip():
    st.info("Please enter a data folder and model checkpoint/folder to begin.")
    st.stop()

CHANNELS = ["TRITC", "FITC", "DAPI"]
OVERLAY_CHANNELS = ["FITC", "TRITC"]
OVERLAY_COLORS = {"FITC": [0, 255, 0], "TRITC": [255, 0, 0]}

st.sidebar.markdown("### Derived paths")
st.sidebar.caption(f"ML folder: `{ML_DIR}`")
st.sidebar.caption(f"Prediction folder: `{PRED_DIR}`")

st.sidebar.markdown("### Folder checks")
st.sidebar.write("Data folder:", "✅" if DATA_DIR.exists() else "❌")
st.sidebar.write("Model path:", "✅" if MODEL_DIR.exists() else "❌")
st.sidebar.write("Prediction folder:", "✅" if PRED_DIR.exists() else "⚠️ run segmentation first")

st.title("RADIX")


# -------------------------
# Segmentation inference
# Rendered before the "no predictions yet" check below so it's reachable
# even on a brand-new dataset with zero predictions.
# -------------------------
with st.expander("Segmentation inference", expanded=False):
    st.markdown("### Run segmentation model")
    st.caption(
        "Runs predict.py on raw per-sample folders and saves full-size predictions for downstream RootLensX analysis."
    )

    seg_input_dir = str(DATA_DIR)
    seg_model_dir = str(MODEL_DIR)
    seg_output_dir = str(ML_DIR)
    seg_pred_dir = str(PRED_DIR)

    st.caption(f"Raw image input: `{seg_input_dir}`")
    st.caption(f"Model checkpoint/folder: `{seg_model_dir}`")
    st.caption(f"Segmentation output: `{seg_output_dir}`")
    st.caption(f"Prediction masks: `{seg_pred_dir}`")

    seg_save_vis = st.checkbox(
        "Save segmentation preview PNGs",
        value=False,
        key="seg_save_vis",
    )

    run_segmentation = st.button("Run segmentation")

    if run_segmentation:
        if not Path(seg_input_dir).exists():
            st.error(f"Input folder does not exist: {seg_input_dir}")
        elif not Path(seg_model_dir).exists():
            st.error(f"Model checkpoint/folder does not exist: {seg_model_dir}")
        else:
            cmd = [
                sys.executable,
                "-u",
                "predict.py",
                "--data-dir", seg_input_dir,
                "--model-dir", seg_model_dir,
                "--output-dir", seg_output_dir,
            ]

            if not seg_save_vis:
                cmd.append("--no-vis")

            with st.expander("Show command", expanded=False):
                st.code(" ".join(cmd))

            progress_bar = st.progress(0)
            progress_text = st.empty()
            log_box = st.empty()

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            total = None
            done = 0
            logs = []
            current_sample = ""

            for line in process.stdout:
                line = line.rstrip()
                logs.append(line)

                if line.startswith("Discovered "):
                    m = re.search(r"Discovered\s+(\d+)\s+samples", line)
                    if m:
                        total = int(m.group(1))
                        progress_text.write(f"Progress: 0 / {total} (0.0%)")

                elif line.startswith("-- "):
                    current_sample = line.replace("-- ", "").strip()
                    done += 1
                    if total:
                        pct = done / total * 100
                        progress_bar.progress(min(done / total, 1.0))
                        progress_text.write(f"Progress: {done} / {total} ({pct:.1f}%)")
                    else:
                        progress_text.write(f"Processed: {done}")

                log_box.info(f"Current sample: {current_sample}" if current_sample else "Starting...")

            return_code = process.wait()

            if return_code == 0:
                st.success("Segmentation finished.")
                st.info(f"Prediction folder: {seg_pred_dir}")
                st.rerun()
            else:
                st.error("Segmentation failed.")
                st.code("\n".join(logs) if logs else "(no output captured)")


# -------------------------
# UI
# -------------------------
samples = sorted(p.stem for p in PRED_DIR.glob("*.npy")) if PRED_DIR.exists() else []

if not samples:
    st.error(
        f"No prediction files found in: {PRED_DIR}\n\n"
        "Run segmentation first using the **Segmentation inference** panel above, "
        "or check that the prediction folder path is correct."
    )
    st.stop()


control_col, preview_col = st.columns([1, 1.4])

with preview_col:
    st.markdown("### Sample selection")

    if samples:
        # Initialize session state keys
        if "sample_index" not in st.session_state:
            st.session_state.sample_index = 0
        
        # Check if we are in "Random Preview" mode
        is_preview_mode = "preview_samples" in st.session_state and st.session_state["preview_samples"]
        current_list = st.session_state["preview_samples"] if is_preview_mode else samples

        # Reset button to exit random preview
        if is_preview_mode:
            if st.button("Clear Random Preview (Show All Samples)"):
                del st.session_state["preview_samples"]
                st.session_state.sample_index = 0
                st.rerun()

        # Ensure index is within bounds of the current list
        st.session_state.sample_index = max(0, min(st.session_state.sample_index, len(current_list) - 1))

        sample = st.selectbox(
            "Sample",
            current_list,
            index=st.session_state.sample_index,
            key="sample_selectbox",
            label_visibility="collapsed",
        )
        st.session_state.sample_index = current_list.index(sample)
        st.caption(f"Viewing {st.session_state.sample_index + 1} / {len(current_list)} samples" + (" (Random Subset)" if is_preview_mode else ""))
    else:
        sample = None
        st.info("No prediction masks found yet. Run segmentation first.")

    # Tissue selection — FITC and TRITC overlays are always shown side by side
    region = st.selectbox(
        "Tissue",
        ["Epidermis", "Exodermis", "Endodermis", "Cortex", "Vascular"],
        key="region_select",
    )

    auto_crop = st.checkbox(
        "Auto-crop previews to selected tissue",
        value=True,
        key="auto_crop_preview",
    )

with control_col:
    with st.expander("Barrier intensity quantification", expanded=False):
        st.markdown("### Barrier intensity quantification")
        st.caption(
            "Measures epidermis/exodermis/cortex/endodermis/vascular TRITC and FITC. "
            "Positive-pixel masks are generated from either raw or normalized images "
            "(depending on threshold mode); intensities are always measured on raw data."
        )

        threshold_mode = st.radio(
            "Threshold mode",
            ["Raw intensity", "Normalized intensity"],
            index=0,
        )

        threshold_dict = {}

        for tissue in BARRIER_TISSUES:
            for ch in ["FITC", "TRITC"]:
                slug = f"{tissue.lower()}_{ch.lower()}"

                if threshold_mode == "Raw intensity":
                    lo, hi = range_slider_with_boxes(
                        f"{tissue} {ch} raw threshold",
                        0, 65535, (600, 65535), step=10,
                        key=f"{slug}_raw",
                    )
                else:
                    lo, hi = range_slider_with_boxes(
                        f"{tissue} {ch} normalized threshold (%)",
                        0, 100, (15, 100), step=1,
                        key=f"{slug}_norm",
                    )
                    lo /= 100
                    hi /= 100

                threshold_dict[(tissue, ch)] = (lo, hi)

        max_preview_samples = max(1, len(samples))

        n_preview_samples = st.number_input(
            "Number of random preview samples",
            min_value=1,
            max_value=max_preview_samples,
            value=min(5, max_preview_samples),
            step=1,
        )

        include_thresholds_output = st.checkbox(
            "Include thresholds in output",
            value=True,
            key="include_thresholds_output",
        )

        preview_random_thresholds = st.button("Preview random samples")
        suggest_thresholds = st.button("Suggest thresholds from all images")
        run_barrier_batch = st.button("Run barrier intensity batch")

        if run_barrier_batch:
            st.write("Running barrier intensity batch...")

            batch_rows = []
            peak_feature_rows = []

            progress = st.progress(0)

            for i, sample_name in enumerate(samples):
                try:
                    raw_i, bio7_pred_i = load_sample(sample_name, DATA_DIR, PRED_DIR, CHANNELS)
                    bio7_i = bio7_pred_i

                    rows = measure_barrier_intensity_batch(
                        raw_i,
                        bio7_i,
                        threshold_mode,
                        threshold_dict,
                    )

                    rows["Sample"] = sample_name
                    rows["Species"] = infer_species(sample_name)
                    rows["Mask_source"] = mask_source
                    # Root geometry + tissue areas, incl. aerenchyma (pixel units)
                    rows.update(measure_root_morphometrics(bio7_i))

                    if include_thresholds_output:
                        rows["Threshold_mode"] = threshold_mode

                        for (tissue_name, channel_name), (low_thr, high_thr) in threshold_dict.items():
                            prefix = f"{tissue_name}_{channel_name}"
                            rows[f"{prefix}_threshold_min"] = low_thr
                            rows[f"{prefix}_threshold_max"] = high_thr

                    batch_rows.append(rows)

                except Exception as e:
                    batch_rows.append({
                        "Sample": sample_name,
                        "Species": infer_species(sample_name),
                        "Error": str(e),
                    })

                progress.progress((i + 1) / len(samples))

            barrier_df = pd.DataFrame(batch_rows)
            st.session_state["barrier_batch_df"] = barrier_df

        if "barrier_batch_df" in st.session_state:
            st.markdown("#### Barrier intensity batch results")
            st.dataframe(st.session_state["barrier_batch_df"])

            csv = st.session_state["barrier_batch_df"].to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download barrier intensity CSV",
                data=csv,
                file_name="barrier_intensity_results.csv",
                mime="text/csv",
            )

        if suggest_thresholds:
            with st.spinner("Scanning all samples for intensity statistics..."):
                suggestions_df = compute_threshold_percentile_suggestions(
                    samples,
                    DATA_DIR,
                    PRED_DIR,
                    CHANNELS,
                    mask_source="Prediction (.npy)",
                    gt_mask_dir=None,
                )

            if suggestions_df.empty:
                errors = getattr(suggestions_df, "_errors", [])
                if mask_source == "Prediction (.npy)":
                    st.error(
                        "No valid samples found. Make sure prediction .npy files exist "
                        f"in: {PRED_DIR}"
                    )
                else:
                    st.error(
                        "No valid samples found. Make sure prediction .npy files exist "
                        f"in: {PRED_DIR}"
                    )
                if errors:
                    with st.expander(f"Show errors ({len(errors)})"):
                        for e in errors[:20]:
                            st.caption(e)
            else:
                errors = getattr(suggestions_df, "_errors", [])
                if errors:
                    st.warning(
                        f"{len(errors)} sample(s) failed to load and were skipped."
                    )
                    with st.expander("Show skipped samples"):
                        for e in errors[:20]:
                            st.caption(e)
                st.session_state["threshold_suggestions_df"] = suggestions_df

        if "threshold_suggestions_df" in st.session_state:
            st.markdown("#### Suggested raw-intensity thresholds")
            st.dataframe(st.session_state["threshold_suggestions_df"])

        if preview_random_thresholds:
            rng = np.random.default_rng()
            st.session_state["preview_samples"] = rng.choice(
                samples,
                size=min(int(n_preview_samples), len(samples)),
                replace=False,
            ).tolist()

# -------------------------
# PROCESS
# -------------------------
analysis_ready = sample is not None

if not analysis_ready:
    st.warning("No sample selected.")
    st.stop()

raw, bio7_pred = load_sample(sample, DATA_DIR, PRED_DIR, CHANNELS)
bio7 = bio7_pred

ch_idx = {"TRITC": 0, "FITC": 1, "DAPI": 2}

mask = bio7[region] > 0
if region == "Cortex":
    mask = mask & ~(bio7["Aerenchyma"] > 0)

_default_preview_range = (0, 65535) if threshold_mode == "Raw intensity" else (0.0, 1.0)

# Wall masks for both channels — the live preview shows FITC and TRITC side by side
ch_imgs_raw = {}
wall_masks = {}
for ch in OVERLAY_CHANNELS:
    ch_img_raw = raw[..., ch_idx[ch]].astype(np.float32)
    ch_img_norm = normalize_channel(ch_img_raw)
    ch_imgs_raw[ch] = ch_img_raw

    preview_min, preview_max = threshold_dict.get((region, ch), _default_preview_range)

    if threshold_mode == "Raw intensity":
        wall_masks[ch] = mask & (ch_img_raw >= preview_min) & (ch_img_raw <= preview_max)
    else:
        wall_masks[ch] = mask & (ch_img_norm >= preview_min) & (ch_img_norm <= preview_max)

preview_samples = st.session_state.get("preview_samples", [])

if len(preview_samples) > 0:
    st.subheader("Random threshold preview")

    for sample_name in preview_samples:
        try:
            raw_i, bio7_i = load_sample(
                sample_name,
                DATA_DIR,
                PRED_DIR,
                CHANNELS,
            )

            mask_i = bio7_i[region] > 0
            if region == "Cortex":
                mask_i = mask_i & ~(bio7_i["Aerenchyma"] > 0)

            rgb_i = to_uint8(normalize_percentile_rgb(raw_i))
            overlays_i = {}

            for ch in OVERLAY_CHANNELS:
                ch_img_raw_i = raw_i[..., ch_idx[ch]].astype(np.float32)
                ch_img_norm_i = normalize_channel(ch_img_raw_i)

                preview_min_i, preview_max_i = threshold_dict.get((region, ch), _default_preview_range)

                if threshold_mode == "Raw intensity":
                    wall_mask_i = mask_i & (ch_img_raw_i >= preview_min_i) & (ch_img_raw_i <= preview_max_i)
                else:
                    wall_mask_i = mask_i & (ch_img_norm_i >= preview_min_i) & (ch_img_norm_i <= preview_max_i)

                overlay_i = rgb_i.copy()
                overlay_i[wall_mask_i] = OVERLAY_COLORS[ch]
                overlays_i[ch] = overlay_i

            if auto_crop:
                _, box_i = crop_to_mask(rgb_i, mask_i)
                y0, y1, x0, x1 = box_i

                rgb_i = rgb_i[y0:y1, x0:x1]
                for ch in OVERLAY_CHANNELS:
                    overlays_i[ch] = overlays_i[ch][y0:y1, x0:x1]

            st.markdown(f"### {sample_name}")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(rgb_i, caption="Raw", width="stretch")
            with c2:
                st.image(overlays_i["FITC"], caption="FITC overlay", width="stretch")
            with c3:
                st.image(overlays_i["TRITC"], caption="TRITC overlay", width="stretch")

        except Exception as e:
            st.warning(f"{sample_name}: {e}")


rgb = to_uint8(normalize_percentile_rgb(raw))
overlays = {}
threshold_only = {}
for ch in OVERLAY_CHANNELS:
    overlay = rgb.copy()
    overlay[wall_masks[ch]] = OVERLAY_COLORS[ch]
    overlays[ch] = overlay

    to_img = np.zeros_like(rgb)
    to_img[wall_masks[ch]] = [255, 255, 255]
    threshold_only[ch] = to_img

if auto_crop:
    _, crop_box = crop_to_mask(rgb, mask)
    y0, y1, x0, x1 = crop_box

    rgb_show = rgb[y0:y1, x0:x1]
    overlay_show = {ch: overlays[ch][y0:y1, x0:x1] for ch in OVERLAY_CHANNELS}
    threshold_only_show = {ch: threshold_only[ch][y0:y1, x0:x1] for ch in OVERLAY_CHANNELS}
else:
    rgb_show = rgb
    overlay_show = overlays
    threshold_only_show = threshold_only

# -------------------------
# VISUALIZATION
# -------------------------

with preview_col:
    st.subheader("Live preview")

    p1, p2, p3 = st.columns(3)

    with p1:
        st.image(
            rgb_show,
            caption="Raw image",
            width="stretch",
        )

    with p2:
        st.image(
            overlay_show["FITC"],
            caption="FITC overlay",
            width="stretch",
        )

    with p3:
        st.image(
            overlay_show["TRITC"],
            caption="TRITC overlay",
            width="stretch",
        )

    q1, q2 = st.columns(2)

    with q1:
        st.image(
            threshold_only_show["FITC"],
            caption="FITC threshold only",
            width="stretch",
        )

    with q2:
        st.image(
            threshold_only_show["TRITC"],
            caption="TRITC threshold only",
            width="stretch",
        )

    show_intensity_hist = st.checkbox(
        "Show intensity distribution (density)",
        value=False,
        key="show_intensity_hist",
    )

    if show_intensity_hist:
        import plotly.graph_objects as go
        from scipy.stats import gaussian_kde

        theme_base = st.get_option("theme.base")
        is_dark = (theme_base == "dark")
        template = "plotly_dark" if is_dark else "plotly_white"
        line_color = "rgba(255,255,255,0.75)" if is_dark else "rgba(0,0,0,0.75)"

        hist_cols = st.columns(2)

        for hist_col, ch in zip(hist_cols, OVERLAY_CHANNELS):
            with hist_col:
                vals_hist = ch_imgs_raw[ch][mask]
                vals_hist = vals_hist[np.isfinite(vals_hist)]

                if vals_hist.size == 0:
                    st.warning(f"No pixels found for selected tissue ({ch}).")
                    continue

                # Subsample if very large (for speed)
                if vals_hist.size > 50000:
                    vals_hist = np.random.choice(vals_hist, 50000, replace=False)

                kde = gaussian_kde(vals_hist)
                x_vals = np.linspace(vals_hist.min(), vals_hist.max(), 500)
                y_vals = kde(x_vals)

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="lines",
                        name="Density",
                        fill="tozeroy",
                    )
                )

                fig.update_layout(
                    template=template,
                    title=f"{sample} — {region} {ch} intensity density",
                    xaxis_title="Raw intensity",
                    xaxis_type="log",
                    yaxis_title="Density",
                )

                # Overlay percentile suggestions
                if "threshold_suggestions_df" in st.session_state:
                    sugg_df = st.session_state["threshold_suggestions_df"]

                    match = sugg_df[
                        (sugg_df["Tissue"] == region) &
                        (sugg_df["Channel"] == ch)
                    ]

                    if len(match) > 0:
                        row = match.iloc[0]

                        for col in match.columns:
                            if col.startswith("p"):
                                try:
                                    p_val = float(row[col])

                                    fig.add_vline(
                                        x=p_val,
                                        line_dash="dash",
                                        line_color=line_color,
                                        line_width=1,
                                        annotation_text=col,
                                        annotation_position="top",
                                        annotation_font_color=line_color,
                                    )
                                except Exception:
                                    pass

                st.plotly_chart(fig, width="stretch", key=f"hist_{ch}")


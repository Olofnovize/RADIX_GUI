"""
src/scale_utils.py

Compute root size and derive scale-invariant defaults for downstream algorithms.

Why this exists
---------------
Different microscopes and species produce images at very different pixel
resolutions in this project (~956 px wide for Tomato up to ~2048 px wide for
Sorghum). Many algorithm parameters in ring_counting.py, blob_counting.py and
related modules are expressed in *absolute pixels* (peak_distance, ring_width,
merge_distance, min_area, ...). A peak_distance tuned for a 2048-px Sorghum
image will under-merge by ~2x on a 956-px Tomato image.

This module computes a single 'cell_scale_px' from the root geometry (or from
microscope physical resolution if known) and provides scale-aware default
parameter dicts that downstream callers can splat into the ring/blob functions.

The downstream functions still accept absolute pixel parameters as before, so
this module is purely additive. Callers that don't use it keep working.

Usage
-----
    from src.scale_utils import compute_root_scale, scaled_ring_defaults

    scale = compute_root_scale(bio7)               # bio7 from load_sample()
    defaults = scaled_ring_defaults(scale)
    stats = ring_peak_count(raw, bio7, tissue="Endodermis",
                            channel="TRITC", **defaults)

Optional: physical scaling when microscope resolution is known
--------------------------------------------------------------
    scale = compute_root_scale(bio7, um_per_pixel=0.32)
"""

from typing import Dict, Any, Optional
import numpy as np
from skimage.measure import label, regionprops


# -----------------------------------------------------------------------------
# Tunable constants
# -----------------------------------------------------------------------------

# Typical cell scale, expressed as a fraction of whole-root diameter.
# Empirical anchor: small cortex cells in cereals are ~15-25 px on a ~1500-px
# root, i.e. ~1.0-1.7% of root diameter. We use 1.2% as a single default.
# This is the one knob to adjust if scale-derived defaults feel systematically
# too tight or too loose across species.
DEFAULT_CELL_FRACTION_OF_ROOT = 0.012

# Typical cortex cell diameter in micrometers. Used only when um_per_pixel is
# provided. Approximate average across cereal and dicot species in this dataset.
DEFAULT_TYPICAL_CELL_UM = 12.0

# Floors so partial / corrupted masks cannot collapse parameters to zero.
MIN_CELL_SCALE_PX = 4.0
MIN_ROOT_DIAMETER_PX = 100.0


# -----------------------------------------------------------------------------
# Core API
# -----------------------------------------------------------------------------

def compute_root_scale(
    bio7: Dict[str, np.ndarray],
    um_per_pixel: Optional[float] = None,
    typical_cell_um: float = DEFAULT_TYPICAL_CELL_UM,
) -> Dict[str, Any]:
    """
    Compute root diameter and a typical cell scale, both in pixels.

    Parameters
    ----------
    bio7
        Tissue mask dict (output of unet_semantic_to_bio7). Must contain
        the "Whole Root" key with a 2D binary / uint8 mask.
    um_per_pixel
        Physical resolution. If provided, cell_scale_px is derived from
        ``typical_cell_um / um_per_pixel`` instead of from root geometry.
        Use this when microscope calibration is known (most accurate).
    typical_cell_um
        Expected typical cortex cell diameter in micrometers. Only used when
        um_per_pixel is provided. Default ~12 um is a rough cereal average.

    Returns
    -------
    dict with keys
        root_diameter_px : float
            Equivalent circular diameter of the largest connected component
            of the whole-root mask, in pixels. Robust to small stray pixels.
        cell_scale_px : float
            Estimated typical cell diameter in pixels.
        source : str
            "physical"        — derived from um_per_pixel
            "root_geometry"   — derived from root diameter
            "fallback_*"      — root mask was missing / empty
        valid : bool
            False if the root mask was empty or had no valid components.
    """
    wr = bio7.get("Whole Root")

    if wr is None or not np.any(wr):
        return _fallback_scale("fallback_empty_mask")

    wr_bin = (wr > 0).astype(np.uint8)

    # Largest connected component handles stray pixels from prediction noise.
    lab = label(wr_bin, connectivity=2)
    props = regionprops(lab)

    if not props:
        return _fallback_scale("fallback_no_props")

    main = max(props, key=lambda p: p.area)
    root_diameter_px = float(max(MIN_ROOT_DIAMETER_PX, main.equivalent_diameter))

    if um_per_pixel is not None and um_per_pixel > 0:
        cell_scale_px = float(max(
            MIN_CELL_SCALE_PX,
            typical_cell_um / um_per_pixel,
        ))
        source = "physical"
    else:
        cell_scale_px = float(max(
            MIN_CELL_SCALE_PX,
            DEFAULT_CELL_FRACTION_OF_ROOT * root_diameter_px,
        ))
        source = "root_geometry"

    return {
        "root_diameter_px": root_diameter_px,
        "cell_scale_px":    cell_scale_px,
        "source":           source,
        "valid":            True,
    }


def scaled_ring_defaults(scale: Dict[str, Any]) -> Dict[str, int]:
    """
    Default kwargs for ring_peak_count() and multi_ring_peak_count(),
    derived from the inferred cell scale.

    The returned dict contains only parameters whose units are pixels.
    Angular params (n_points, smooth_sigma) and unitless params
    (peak_prominence after normalization, local_contrast_threshold) are
    intentionally excluded — keep their existing function defaults.

    Returns
    -------
    dict
        peak_distance, ring_width, merge_distance, peak_width
    """
    cs = float(scale["cell_scale_px"])

    return {
        "peak_distance":  int(max(3, round(0.45 * cs))),
        "ring_width":     int(max(2, round(0.30 * cs))),
        "merge_distance": int(max(4, round(1.10 * cs))),
        "peak_width":     int(max(1, round(0.15 * cs))),
    }


def scaled_blob_defaults(scale: Dict[str, Any]) -> Dict[str, int]:
    """
    Default kwargs for estimate_lumen() and similar blob-area functions,
    derived from the inferred cell scale.

    Returns
    -------
    dict
        min_area, max_area, min_distance
        Areas are in pixels squared. min_distance is the minimum spacing
        between watershed markers in pixels.
    """
    cs = float(scale["cell_scale_px"])
    cell_area_px = np.pi * (cs / 2.0) ** 2

    return {
        # ~ half a typical cell area; reject sub-cellular specks
        "min_area":     int(max(20,   round(0.5 * cell_area_px))),
        # ~ 80x typical cell area; allows large xylem lumens but rejects
        # whole-tissue blobs
        "max_area":     int(max(2000, round(80.0 * cell_area_px))),
        # min spacing between watershed markers
        "min_distance": int(max(3,    round(0.5 * cs))),
    }


def describe_scale(scale: Dict[str, Any]) -> str:
    """
    Compact one-line summary, useful for a GUI sidebar caption.

    Example
    -------
    >>> describe_scale(scale)
    'Root diameter: 1480 px  •  Cell scale: 17.8 px  •  Source: root_geometry'
    """
    return (
        f"Root diameter: {scale['root_diameter_px']:.0f} px  •  "
        f"Cell scale: {scale['cell_scale_px']:.1f} px  •  "
        f"Source: {scale['source']}"
    )


# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------

def _fallback_scale(reason: str) -> Dict[str, Any]:
    return {
        "root_diameter_px": MIN_ROOT_DIAMETER_PX,
        "cell_scale_px":    MIN_CELL_SCALE_PX,
        "source":           reason,
        "valid":            False,
    }

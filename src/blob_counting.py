import streamlit as st
from scipy import ndimage as ndi
from skimage import morphology, feature, segmentation, measure
import numpy as np

from .scale_utils import compute_root_scale, scaled_blob_defaults


def estimate_lumen(
    tissue_mask,
    wall_mask,
    min_area=None,
    max_area=None,
    scale=None,
    show_debug=False,
):
    """Estimate lumen-like compartments inside a tissue mask via watershed.

    If ``min_area`` / ``max_area`` are not given they are derived from
    ``scale`` via :func:`scaled_blob_defaults`. If ``scale`` is also missing
    we fall back to deriving it from the tissue mask itself, which is less
    accurate than passing a ``scale`` derived from the whole-root mask
    (use :func:`compute_root_scale(bio7)` once per sample at the call site).
    """

    # Resolve scale-dependent defaults. Explicit values from caller win.
    if scale is None:
        # Fallback: use the tissue mask as a proxy for whole root.
        scale = compute_root_scale({"Whole Root": tissue_mask})
    _defaults = scaled_blob_defaults(scale)
    if min_area is None:
        min_area = _defaults["min_area"]
    if max_area is None:
        max_area = _defaults["max_area"]
    min_distance = _defaults["min_distance"]

    lumen0 = tissue_mask & (~wall_mask)

    if show_debug:
        st.write("BLOB DEBUG 1 tissue px:", int(tissue_mask.sum()))
        st.write("BLOB DEBUG 2 wall px:", int(wall_mask.sum()))
        st.write("BLOB DEBUG 3 raw lumen px:", int(lumen0.sum()))
        st.write(
            "BLOB DEBUG scale:",
            f"min_area={min_area} max_area={max_area} min_distance={min_distance}",
        )

    lumen = morphology.remove_small_objects(lumen0, min_size=min_area)

    if show_debug:
        st.write("BLOB DEBUG 4 lumen after min_area px:", int(lumen.sum()))

    distance = ndi.distance_transform_edt(lumen)

    coords = feature.peak_local_max(
        distance,
        labels=lumen,
        min_distance=min_distance,
        exclude_border=False,
    )

    if show_debug:
        st.write("BLOB DEBUG 5 watershed markers:", len(coords))

    markers = np.zeros_like(distance, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    labels = segmentation.watershed(
        -distance,
        markers,
        mask=lumen,
    )

    props = measure.regionprops(labels)

    if show_debug:
        st.write("BLOB DEBUG 6 watershed props before filtering:", len(props))
        st.write("BLOB DEBUG 7 prop areas:", [int(p.area) for p in props[:20]])

    cleaned = np.zeros_like(labels)
    new_label = 1

    valid_props = [p for p in props if p.area >= min_area]

    if show_debug:
        st.write("BLOB DEBUG 8 valid props after filtering:", len(valid_props))

    for p in valid_props:
        cleaned[labels == p.label] = new_label
        new_label += 1

    return cleaned
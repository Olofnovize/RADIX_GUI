import streamlit as st

def slider_with_box(label, min_value, max_value, value, step=1, key=None):
    """Synchronized Streamlit slider + number box."""
    if key is None:
        key = label.lower().replace(" ", "_").replace("/", "_")

    main_key = key
    slider_key = f"{key}_slider"
    box_key = f"{key}_box"

    if main_key not in st.session_state:
        st.session_state[main_key] = value

    def update_from_slider():
        st.session_state[main_key] = st.session_state[slider_key]

    def update_from_box():
        st.session_state[main_key] = st.session_state[box_key]

    col_slider, col_box = st.columns([3, 1])

    with col_slider:
        st.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            value=st.session_state[main_key],
            step=step,
            key=slider_key,
            on_change=update_from_slider,
        )

    with col_box:
        st.number_input(
            "Value",
            min_value=min_value,
            max_value=max_value,
            value=st.session_state[main_key],
            step=step,
            key=box_key,
            label_visibility="collapsed",
            on_change=update_from_box,
        )

    return st.session_state[main_key]


def range_slider_with_boxes(label, min_value, max_value, value, step=1, key=None):
    """Range slider synchronized with MIN and MAX number boxes."""
    if key is None:
        key = label.lower().replace(" ", "_").replace("/", "_")

    range_key = f"{key}_range"
    min_key = f"{key}_min"
    max_key = f"{key}_max"

    if range_key not in st.session_state:
        st.session_state[range_key] = value
    if min_key not in st.session_state:
        st.session_state[min_key] = value[0]
    if max_key not in st.session_state:
        st.session_state[max_key] = value[1]

    def update_from_slider():
        low, high = st.session_state[range_key]
        st.session_state[min_key] = low
        st.session_state[max_key] = high

    def update_from_boxes():
        low = st.session_state[min_key]
        high = st.session_state[max_key]

        low = max(min_value, min(low, max_value))
        high = max(min_value, min(high, max_value))

        if low > high:
            low, high = high, low

        st.session_state[min_key] = low
        st.session_state[max_key] = high
        st.session_state[range_key] = (low, high)

    col_min, col_slider, col_max = st.columns([1, 3, 1])

    with col_min:
        st.number_input(
            "Min",
            min_value=min_value,
            max_value=max_value,
            step=step,
            key=min_key,
            label_visibility="collapsed",
            on_change=update_from_boxes,
        )

    with col_slider:
        st.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            step=step,
            key=range_key,
            on_change=update_from_slider,
        )

    with col_max:
        st.number_input(
            "Max",
            min_value=min_value,
            max_value=max_value,
            step=step,
            key=max_key,
            label_visibility="collapsed",
            on_change=update_from_boxes,
        )

    return st.session_state[range_key]
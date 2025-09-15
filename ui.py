import streamlit as st

def render_field(key, meta, current):
    label = meta.get("label", key)
    help_text = meta.get("help")
    widget = meta.get("widget")  # slider / number / radio / select / checkbox
    dtype = meta.get("type")     # int / float / str / bool

    # 1) SLIDER (numbers in a range)
    if widget == "slider":
        min_v = meta.get("min", 0)
        max_v = meta.get("max", 100)
        step = meta.get("step", 1 if dtype == "int" else 0.1)
        if dtype == "int":
            return st.slider(label, int(min_v), int(max_v), int(current), help=help_text)
        else:  # float
            return st.slider(label, float(min_v), float(max_v), float(current), step=float(step), help=help_text)

    # 2) NUMBER INPUT (numbers with arrows)
    if widget == "number":
        min_v = meta.get("min", 0)
        max_v = meta.get("max", 100)
        if dtype == "int":
            return st.number_input(label, min_value=int(min_v), max_value=int(max_v), value=int(current), step=1, help=help_text)
        else:  # float
            step = float(meta.get("step", 0.1))
            return st.number_input(label, min_value=float(min_v), max_value=float(max_v), value=float(current), step=step, help=help_text)

    # 3) RADIO (few options, all visible)
    if widget == "radio":
        options = meta.get("options", [])
        index = options.index(current) if current in options else 0
        return st.radio(label, options=options, index=index, horizontal=True, help=help_text)

    # 4) SELECT (drop‑down)
    if widget == "select":
        options = meta.get("options", [])
        index = options.index(current) if current in options else 0
        return st.selectbox(label, options=options, index=index, help=help_text)

    # 5) CHECKBOX (true/false)
    if widget == "checkbox":
        return st.checkbox(label, value=bool(current), help=help_text)

    # Unknown widget
    st.warning(f"Unsupported widget for field '{key}'")
    return current
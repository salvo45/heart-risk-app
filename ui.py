import streamlit as st

def render_footer(name: str, linkedin: str | None = None, github: str | None = None):
    parts = [f"Progetto di **{name}**"]
    if linkedin:
        parts.append(f"[LinkedIn]({linkedin})")
    if github:
        parts.append(f"[GitHub]({github})")
    st.divider()
    st.caption(" · ".join(parts))
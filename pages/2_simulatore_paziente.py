import streamlit as st
st.title("🩺 Simulatore Paziente")
st.write("""In questa pagina l'utente inserirà i propri valori (età, pressione,
colesterolo, ecc.)
 e otterrà:
 - una percentuale di rischio,
 - un esito basato su soglia regolabile,
 - una spiegazione breve sui fattori che pesano di più.

""")

from ui import render_footer, render_field
from static import FEATURES, DEFAULT_PROFILE, ORDER
if "patient" not in st.session_state:
 st.session_state["patient"] = DEFAULT_PROFILE.copy()
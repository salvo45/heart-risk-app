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

 st.title("🩺 Patient Simulator")
st.subheader("Patient — Enter your values")
# New values will be collected here
updated_values = {}
# Draw each field directly (no forms/containers)
for key in ORDER:
 meta = FEATURES[key]
 current_value = st.session_state["patient"].get(key, meta.get("value"))
 new_value = render_field(key, meta, current_value)
 updated_values[key] = new_value
st.divider()
save = st.button("Save data")
# What happens when button is pressed
if save:
 st.session_state["patient"] = updated_values
 st.success("Patient data saved. (Risk calculation comes in the next step.)")
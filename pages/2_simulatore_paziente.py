from ml import predict_patient_label_and_probability
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

st.subheader("Threshold and model result")



threshold = st.slider(
    "Choose a threshold (%)",
    min_value=0,
    max_value=100,
    value=20,
    step=1,
)

# Use the trained model if available
if "model" not in st.session_state:
    st.warning("No trained model in memory. Go to **📊 Data & Model** and click **Train model**.")
else:
    trained_model = st.session_state["model"]
    current_patient = st.session_state["patient"]

    predicted_label, probability_of_disease = predict_patient_label_and_probability(
        trained_model, current_patient
    )

    probability_percent = int(round(probability_of_disease * 100))
    result_text = "ALERT" if probability_percent >= threshold else "LOW RISK"

    left_column, right_column = st.columns(2)
    left_column.metric("Model probability", f"{probability_percent}%")
    right_column.metric("Result at threshold", result_text)

    st.caption(
        "This result comes from the trained Logistic Regression. "
        "Move the threshold slider to see how the label changes."
    )
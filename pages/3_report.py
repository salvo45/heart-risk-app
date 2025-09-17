import json
from datetime import datetime

import pandas as pd
import streamlit as st

st.title("📄 Report")
st.write(
 """
 Qui verrà generato un riepilogo scaricabile dei risultati:
 - input inseriti,
 - percentuale ed esito,
 - data/ora e versione del modello.

 """
)

# pages/4_📄_Report.py — minimal download of prediction recap (inputs + label)



from ml import predict_patient_label_and_probability

st.title("📄 Report — Prediction recap")

# 1) Preconditions: we need a saved patient and a trained model
if "patient" not in st.session_state:
    st.warning("No patient data in memory. Go to 🩺 Patient Simulator, fill the form and click 'Save data'.")
    st.stop()

if "model" not in st.session_state:
    st.warning("No trained model in memory. Go to 📊 Data & Model and click 'Train model'.")
    st.stop()

# 2) Compute the current label using the trained model (binary)
patient_dict = st.session_state["patient"]
trained_model = st.session_state["model"]

predicted_label, probability_of_disease = predict_patient_label_and_probability(trained_model, patient_dict)
label_text = "ALERT" if predicted_label == 1 else "LOW RISK"

# 3) Build a minimal recap object (only inputs + label)
recap = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "label": label_text,
    "inputs": patient_dict,
}

st.subheader("Preview")
st.json(recap)

# 4) Prepare JSON and CSV for download
json_bytes = json.dumps(recap, indent=2).encode("utf-8")

row_for_csv = {**patient_dict, "label": label_text}
dataframe_for_csv = pd.DataFrame([row_for_csv])
csv_bytes = dataframe_for_csv.to_csv(index=False).encode("utf-8")

st.subheader("Download")
st.download_button(
    label="Download prediction_recap.json",
    data=json_bytes,
    file_name="prediction_recap.json",
    mime="application/json",
)

st.download_button(
    label="Download prediction_recap.csv",
    data=csv_bytes,
    file_name="prediction_recap.csv",
    mime="text/csv",
)

st.caption(
    "The recap includes only the patient inputs and the obtained label, as requested for the MVP."
)


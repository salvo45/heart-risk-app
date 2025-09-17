import streamlit as st
st.title("Dati e Modello - panoramica")
st.write("in questa pagina verranno mostrati: " \
          "il dataset utilizzato (anteprima descrizione colonne," \
          " i passaggi di pre-processing (pulizia, encoding, scaling) il modello scelto per la classificazione(baseline))")

import streamlit as st
from ml import train_logistic_regression_model



st.write(
    "Fai clic sul pulsante per addestrare una **regressione** logistica **binaria** sul dataset OpenML del cuore. "
    "Il modello addestrato verrà memorizzato in memoria (stato della sessione) e utilizzato dalla pagina Simulator."
)

train_button_clicked = st.button("Train model")

if train_button_clicked:
    try:
        st.info("⏳ Addestramento in corso... Questo potrebbe richiedere alcuni secondi la prima volta (download del dataset).")
        trained_model = train_logistic_regression_model()
        st.session_state["model"] = trained_model  # store for other pages
        st.success("✅ Modello addestrato e memorizzato nella memoria.")
        st.caption("LogisticRegression con impostazioni predefinite. Nessuna preelaborazione, nessun pipeline.")
    except Exception as error:
        st.error(f"Addestramento fallito. Dettagli: {error}")

# Status indicator (so students know if a model is available)
if "model" in st.session_state:
    st.success("Un modello addestrato è disponibile in memoria ed è pronto per essere utilizzato nella pagina Simulator.")
else:
    st.warning("Nessun modello in memoria ancora. Fai clic su 'Addestra modello' per crearne uno.")

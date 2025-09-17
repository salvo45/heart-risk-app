import streamlit as st

st.set_page_config(page_title="Heart Risk Prediction App", page_icon="❤️", layout="centered")

st.title("Heart Risk Prediction App")
st.write(" web-app didattica per stimare una percentuale di rischio cardiovascolare a partire da alcuni parametri clinici basilari.")


from ui import render_footer
render_footer("Salvatore Cammarata",github="https://github.com/SalvoCammarata")

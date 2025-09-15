FEATURES = {
    # 1) Età (anni)
    "age": {
        "label": "Età",
        "widget": "slider",
        "type": "int",
        "min": 18,
        "max": 100,
        "value": 50,
        "help": "Anni (18–100)"
    },
    # 2) Sesso (F/M)
    "sex": {
        "label": "Sesso",
        "widget": "radio",
        "type": "str",
        "options": ["F", "M"],
        "value": "M",
        "help": "Sesso biologico"
    },
    # 3) Tipo di dolore toracico (4 categorie)
    "cp": {
        "label": "Dolore toracico",
        "widget": "select",
        "type": "str",
        "options": [
            "tipico", "atipico", "non-anginoso", "asintomatico"
        ],
        "value": "non-anginoso",
        "help": "Tipo di dolore riportato"
    },
    # 4) Pressione a riposo (mmHg)
    "trestbps": {
        "label": "Pressione a riposo (mmHg)",
        "widget": "slider",
        "type": "int",
        "min": 80,
        "max": 200,
        "value": 130,
        "help": "Valore sistolico a riposo"
    },
    # 5) Colesterolo totale (mg/dL)
    "chol": {
        "label": "Colesterolo totale (mg/dL)",
        "widget": "slider",
        "type": "int",
        "min": 100,
        "max": 600,
        "value": 220,
        "help": "mg/dL (100–600)"
    },
    # 6) Glicemia a digiuno alta (≥120 mg/dL)
    "fbs": {
        "label": "Glicemia a digiuno alta (≥120 mg/dL)",
        "widget": "checkbox",
        "type": "bool",
        "value": False,
        "help": "Spunta se la glicemia a digiuno è ≥120 mg/dL"
    },
    # 7) ECG a riposo (3 categorie)
    "restecg": {
        "label": "ECG a riposo",
        "widget": "select",
        "type": "str",
        "options": [
            "normale", "anomalia ST–T", "ipertrofia ventricolare sinistra"
        ],
        "value": "normale",
        "help": "Risultato ECG a riposo"
    },
    # 8) Frequenza cardiaca massima (bpm)
    "thalach": {
        "label": "Frequenza massima (bpm)",
        "widget": "slider",
        "type": "int",
        "min": 60,
        "max": 220,
        "value": 150,
        "help": "Battiti per minuto durante sforzo"
    },
    # 9) Angina da sforzo (sì/no)
    "exang": {
        "label": "Angina da sforzo",
        "widget": "checkbox",
        "type": "bool",
        "value": False,
        "help": "Spunta se presente durante sforzo"
    },
    # 10) Oldpeak (depressione ST)
    "oldpeak": {
        "label": "Oldpeak (depressione ST)",
        "widget": "number",
        "type": "float",
        "min": 0.0,
        "max": 6.0,
        "step": 0.1,
        "value": 1.0,
        "help": "Differenza ST da riposo a sforzo (0.0–6.0)"
    },
    # 11) Pendenza del tratto ST (slope)
    "slope": {
        "label": "Pendenza ST (slope)",
        "widget": "select",
        "type": "str",
        "options": ["in salita", "piatta", "in discesa"],
        "value": "piatta",
        "help": "Andamento del tratto ST con sforzo"
    },
}
# Utilità: valori di default pronti per inizializzare il form
DEFAULT_PROFILE = {
    "age": 50,
    "sex": "M",
    "cp": "non-anginoso",
    "trestbps": 130,
    "chol": 220,
    "fbs": False,
    "restecg": "normale",
    "thalach": 150,
    "exang": False,
    "oldpeak": 1.0,
    "slope": "piatta",
}
# Display order for the 11 fields
ORDER = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang", "oldpeak", "slope"
]
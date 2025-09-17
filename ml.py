# ml.py — minimal ML using a local CSV (heart.csv)


from typing import List, Tuple, Dict, Any
import math
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

# The same order of features used in the UI (Patient Simulator)
FEATURE_ORDER: List[str] = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang", "oldpeak", "slope"
]

# -------------------------------
# 1) Encode patient dict → numeric list (matches training columns)
# -------------------------------

def encode_patient_to_numeric_list(patient: Dict[str, Any]) -> List[float]:
    """Convert a patient (dict from the UI) into a list of numbers matching FEATURE_ORDER."""
    # sex: F=0, M=1
    sex_numeric = 1 if patient["sex"] == "M" else 0

    # cp (chest pain): 1=typical, 2=atypical, 3=non-anginal, 4=asymptomatic
    if patient["cp"] == "tipico":
        cp_numeric = 1
    elif patient["cp"] == "atipico":
        cp_numeric = 2
    elif patient["cp"] == "non-anginoso":
        cp_numeric = 3
    else:  # "asintomatico"
        cp_numeric = 4

    # restecg: 0=normal, 1=ST-T abnormality, 2=left ventricular hypertrophy
    if patient["restecg"] == "normale":
        restecg_numeric = 0
    elif patient["restecg"] == "anomalia ST–T":
        restecg_numeric = 1
    else:  # "ipertrofia ventricolare sinistra"
        restecg_numeric = 2

    # slope: 1=upsloping, 2=flat, 3=downsloping
    if patient["slope"] == "in salita":
        slope_numeric = 1
    elif patient["slope"] == "piatta":
        slope_numeric = 2
    else:  # "in discesa"
        slope_numeric = 3

    # booleans → 0/1
    fasting_blood_sugar_numeric = 1 if patient["fbs"] else 0
    exercise_induced_angina_numeric = 1 if patient["exang"] else 0

    # numeric as float
    age_numeric = float(patient["age"])
    resting_blood_pressure_numeric = float(patient["trestbps"])
    cholesterol_numeric = float(patient["chol"])
    max_heart_rate_numeric = float(patient["thalach"])
    oldpeak_numeric = float(patient["oldpeak"])

    return [
        age_numeric, sex_numeric, cp_numeric, resting_blood_pressure_numeric, cholesterol_numeric,
        fasting_blood_sugar_numeric, restecg_numeric, max_heart_rate_numeric,
        exercise_induced_angina_numeric, oldpeak_numeric, slope_numeric
    ]

# -------------------------------
# 2) Load CSV → build X (features) and y (labels)
# -------------------------------

def load_heart_dataframe() -> pd.DataFrame:
    """Read heart.csv from the project root and return a DataFrame
    containing the 11 features we use + a binary target column.
    """
    csv_path = Path(__file__).resolve().parent / "heart.csv"
    if not csv_path.exists():
        raise FileNotFoundError("File 'heart.csv' not found in the project root.")

    dataframe = pd.read_csv(csv_path)

    # normalize the target column name to 'target'
    if "target" not in dataframe.columns and "num" in dataframe.columns:
        dataframe = dataframe.rename(columns={"num": "target"})

    # keep only the columns we need (if present)
    columns_we_need = FEATURE_ORDER + ["target"]
    existing_columns = []
    for name in columns_we_need:
        if name in dataframe.columns:
            existing_columns.append(name)
    dataframe = dataframe[existing_columns].copy()

    return dataframe


def build_feature_rows_and_target_values(dataframe: pd.DataFrame) -> Tuple[List[List[float]], List[int]]:
    """Create plain Python lists for features (X) and labels (y).
    Skip rows with missing values. Convert the target to 0/1.
    """
    feature_rows: List[List[float]] = []
    target_values: List[int] = []

    # if the CSV uses 'num' 0..4 and we renamed to 'target', convert to binary 0/1
    # if it already has 0/1 values, the conversion keeps them as they are
    for index, row in dataframe.iterrows():
        # skip rows with any missing value in our columns
        has_missing = False
        for column_name in FEATURE_ORDER + ["target"]:
            if column_name not in dataframe.columns:
                has_missing = True
                break
            if pd.isna(row[column_name]):
                has_missing = True
                break
        if has_missing:
            continue

        # extract features in the exact order
        age_value = float(row["age"])
        sex_value = int(row["sex"])
        cp_value = int(row["cp"])
        trestbps_value = float(row["trestbps"])
        chol_value = float(row["chol"])
        fbs_value = int(row["fbs"])
        restecg_value = int(row["restecg"])
        thalach_value = float(row["thalach"])
        exang_value = int(row["exang"])
        oldpeak_value = float(row["oldpeak"])
        slope_value = int(row["slope"])

        feature_rows.append([
            age_value, sex_value, cp_value, trestbps_value, chol_value,
            fbs_value, restecg_value, thalach_value, exang_value, oldpeak_value, slope_value
        ])

        # convert target to binary: 1 = disease present, 0 = no disease
        raw_target = row["target"]
        integer_target = int(float(raw_target))
        binary_target = 0 if integer_target == 0 else 1
        target_values.append(binary_target)

    return feature_rows, target_values

# -------------------------------
# 3) Train and Predict (binary)
# -------------------------------

def train_logistic_regression_model() -> LogisticRegression:
    """Load data from heart.csv, build features/labels, train a Logistic Regression.
    No preprocessing, no pipeline. Just fit the model.
    """
    dataframe = load_heart_dataframe()
    feature_rows, target_values = build_feature_rows_and_target_values(dataframe)

    logistic_regression_model = LogisticRegression(max_iter=1000)
    logistic_regression_model.fit(feature_rows, target_values)
    return logistic_regression_model


def predict_patient_label_and_probability(trained_model: LogisticRegression, patient: Dict[str, Any]) -> Tuple[int, float]:
    """Given a trained model and a patient dict, return (label_0_or_1, probability_of_disease_0_to_1).
    Uses predict_proba when available; otherwise falls back to a simple sigmoid on the decision score.
    """
    numeric_features = [encode_patient_to_numeric_list(patient)]

    if hasattr(trained_model, "predict_proba"):
        probability_of_disease = float(trained_model.predict_proba(numeric_features)[0][1])
    else:
        decision_score = float(trained_model.decision_function(numeric_features)[0])
        probability_of_disease = 1.0 / (1.0 + math.exp(-decision_score))

    predicted_label = 1 if probability_of_disease >= 0.5 else 0
    return predicted_label, probability_of_disease


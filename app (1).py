# Author: Hanife Kaptan
# Versions: python==3.11.2, xgboost==2.1.1, streamlit==1.36.0, scikit-learn==1.4.2, pandas==2.2.3

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier 
import streamlit as st

df = pd.read_csv("akcigerKanseri.csv")

X = df.drop("lung_cancer", axis=1)
y = df["lung_cancer"]

preprocess = ColumnTransformer(
    transformers = [("cat", OneHotEncoder(), ["gender"]), ("num", (StandardScaler()), ["age"])], remainder = "passthrough"
)

my_model = XGBClassifier()

pipe = Pipeline(steps=[("preprocessor", preprocess), ("model", my_model)])
pipe.fit(X, y)

def lung_cancer(gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
                chronic_disease, fatigue, allergy, wheezing, alcohol_consuming,
                coughing, shortness_of_breath, swallowing_difficulty, chest_pain):
    input_data = pd.DataFrame({"gender": [gender],
                               "age": [age],
                               "smoking": [smoking],
                               "yellow_fingers": [yellow_fingers],
                               "anxiety": [anxiety],
                               "peer_pressure": [peer_pressure],
                               "chronic_disease": [chronic_disease],
                               "fatigue": [fatigue],
                               "allergy": [allergy],
                               "wheezing": [wheezing],
                               "alcohol_consuming": [alcohol_consuming],
                               "coughing": [coughing],
                               "shortness_of_breath": [shortness_of_breath],
                               "swallowing_difficulty": [swallowing_difficulty],
                               "chest_pain": [chest_pain]})
    prediction = pipe.predict(input_data)[0]
    return prediction

st.title("Akciğer Kanseri Tespiti :hospital:: @hanifekaptan")
st.write("Kendinizle ilgili doğru seçenekleri seçiniz.")
gender = st.radio("Gender", ["Male", "Female"]) # male ve female 1 ve 0 değerlerine dönüştürülecek
age = st.number_input("Age", 0, 100)
smoking = st.radio("Smoking", [True, False])
yellow_fingers = st.radio("Yellow Fingers", [True, False])
anxiety = st.radio("Anxiety", [True, False])
peer_pressure = st.radio("Peer Pressure", [True, False])
chronic_disease = st.radio("Chronic Disease", [True, False])
fatigue = st.radio("Fatigue", [True, False])
allergy = st.radio("Allergy", [True, False])
wheezing = st.radio("Wheezing", [True, False])
alcohol_consuming = st.radio("Alcohol Consuming", [True, False])
coughing = st.radio("Coughing", [True, False])
shortness_of_breath = st.radio("Shortness of Breath", [True, False])
swallowing_difficulty = st.radio("Swallowing Difficulty", [True, False])
chest_pain = st.radio("Chest Pain", [True, False])

if st.button("Predict"):
    pred = lung_cancer(gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
                       chronic_disease, fatigue, allergy, wheezing, alcohol_consuming,
                       coughing, shortness_of_breath, swallowing_difficulty, chest_pain)
    if pred == 1:
        st.write("Result: Positive")
    elif pred == 0:
        st.write("Result: Negative")
import streamlit as st
import numpy as np
import pickle

model, scaler = pickle.load(open('model.pkl', 'rb'))

st.title("💳 Fraud Detection")

inputs = []
for i in range(1, 29):
    val = st.number_input(f'V{i}', value=0.0)
    inputs.append(val)

amount = st.number_input("Amount", value=0.0)
inputs.append(amount)

if st.button("Predict"):
    data = np.array(inputs).reshape(1, -1)
    data = scaler.transform(data)
    
    pred = model.predict(data)[0]

    if pred == 1:
        st.error("Fraud!")
    else:
        st.success("Normal")

import streamlit as st
import joblib as jb
import numpy as np
from sklearn.linear_model import LinearRegression as lr

model = jb.load("model.joblib")
st.markdown("# E-commerce Expense Predictor")
st.markdown("---")

sess =  st.number_input("Average session length")
app_time = st.number_input("Time on app")
web_time =  st.number_input("Time on Website")
mem_length = st.number_input("Length of Membership")

if st.button("Predict"):
    sample = np.array([sess, app_time, web_time, mem_length]).reshape(1,-1)
    prediction = model.predict(sample)[0]
    prediction = f'${prediction: .2f}'
    st.info(f'This customer will likely spend around {prediction}')





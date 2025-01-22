# Import Libraries
import streamlit as st
import pickle
import numpy as np

#load the saved model
model=pickle.load(open(r"C:\Users\sangu\A vs code\ml project\logistic_regression.pkl",'rb'))

#set the titel of the streamlit app
st.title("Logistic Regression Model:--")
age = st.number_input("Enter your age ")
salary = st.number_input("Enter your salary")

temp = np.array([[age,salary]])
if st.button('predict'):
    st.write(model.predict(temp))

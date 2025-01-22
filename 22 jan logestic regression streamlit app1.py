# Import Libraries
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Streamlit Title
st.title("Logistic Regression Model:--")

# Import dataset
dataset = pd.read_csv(r"C:\Users\sangu\Downloads\logit classification.csv")

# Upload dataset using Streamlit
uploaded_file = st.file_uploader(r"C:\Users\sangu\Downloads\logit classification.csv", type=["csv"])

if uploaded_file is not None:
    # Read the dataset
    dataset = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(dataset.head())

    # Split with Iv and Dv
    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, -1].values

    # 75-25 Train test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Scale the Data
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Train the LogisticRegression model on the training set
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    # Predicting the test set result
    y_pred = classifier.predict(X_test)

    # Display Results
    st.subheader("Results")

    # Making the confusion matrix now
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)

    # MODEL ACCURACY
    from sklearn.metrics import accuracy_score
    ac = accuracy_score(y_test, y_pred)
    st.write("Accuracy Score:", ac)

    # Classification report
    from sklearn.metrics import classification_report
    cr = classification_report(y_test, y_pred)
    st.write("Classification Report:")
    st.text(cr)

    # Training Accuracy
    bias = classifier.score(X_train, y_train)
    st.write("Training Accuracy (Bias):", bias)

    # Testing Accuracy
    variance = classifier.score(X_test, y_test)
    st.write("Testing Accuracy (Variance):", variance)

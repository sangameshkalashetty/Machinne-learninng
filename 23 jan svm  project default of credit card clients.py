# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Importing the dataset
dataset = pd.read_excel(r"C:\Users\sangu\Downloads\23rd - SVM\21st - SVM\SVM\project\default of credit card clients.xls")

#Splitting the dataset into the Training set and Test set
dataset.columns = dataset.iloc[0]
dataset = dataset[1:]  
dataset.reset_index(drop=True, inplace=True)


# Defining features (X) and target (y)
X = dataset.drop(columns= 'default payment next month').values
y = dataset.iloc[:, -1].values  # Target variable
y = y.astype('int')


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the SVM model on the Training set
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
classifier = SVC()  # You can use different kernels like 'rbf' or 'poly'
classifier.fit(X_train, y_train)
dt.fit(X_train,y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Model Accuracy
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print("Accuracy:", ac)

# Classification Report
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print("Classification Report:")
print(cr)

# Bias (Training Accuracy)
bias = classifier.score(X_train, y_train)
print("Training Accuracy (Bias):", bias)

# Variance (Testing Accuracy)
variance = classifier.score(X_test, y_test)
print("Testing Accuracy (Variance):", variance)

from sklearn.metrics import accuracy_score 
ac = accuracy_score(y_test, y_pred)
print(ac)  

# Pickling the models and scaler
with open('svm_classifier.pkl', 'wb') as svm_file:
    pickle.dump(classifier, svm_file)

with open('decision_tree_classifier.pkl', 'wb') as dt_file:
    pickle.dump(dt, dt_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(sc, scaler_file)

print("Models and scaler saved successfully.")


#Import  Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset = pd.read_csv(r"C:\Users\sangu\Downloads\logit classification.csv")

#Split with Iv and Dv
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values

#75-25 Train test split
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20,random_state=0)

#Scale the Data
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Train the LogisticsRegression model on the training set
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)

#predicting the test set result
y_pred=classifier.predict(X_test)

#Making the confusin matrix now
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#MODEL ACCURACY
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

#Classification report
from sklearn.metrics import classification_report
cr=classification_report(y_test, y_pred)
cr

#Training Accuracy
bias=classifier.score(X_train, y_train)
print(bias)

#Testing Accuracy
variance=classifier.score(X_test,y_test)
print(variance)
                            Feature Scaling
                            
dataset1= pd.read_csv(r"C:\Users\sangu\Downloads\Future prediction1.csv")

d2=dataset1.copy()

dataset1 = dataset1.iloc[:, [2, 3]].values

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
M=sc.fit_transform(dataset1)

y_pred1= pd.DataFrame()

d2 ['y_pred1']=classifier.predict(M)
d2.to_csv('pred_mpdel.csv')

#to get the path

import os
os.getcwd()




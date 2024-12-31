import numpy as np 	
import matplotlib.pyplot as plt
import pandas as pd	
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

dataset = pd.read_csv(r'C:\Users\sangu\Downloads\Salary_Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

plt.scatter(X_train, y_train, color='red') 
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, y_test, color='red') 
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

y_12 = regressor.predict([[12]])
y_20 = regressor.predict([[20]])

print(f"Predicted salary for 12 years of experience: ${y_12[0]:,.2f}")
print(f"Predicted salary for 20 years of experience: ${y_20[0]:,.2f}")

bias = regressor.score(X_train, y_train)
variance = regressor.score(X_test, y_test)
train_mse = mean_squared_error(y_train, regressor.predict(X_train))
test_mse = mean_squared_error(y_test, y_pred)

print(f"Training Score (R^2): {bias:.2f}")
print(f"Testing Score (R^2): {variance:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")

filename = 'linear_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print("Model has been pickled and saved as linear_regression_model.pkl")

dataset.mean()

dataset['Salary'].mean()
dataset.describe()

dataset.corr()

dataset.skew()

dataset['Salary'].skew()

dataset.sem()

#z score
import scipy.stats as stats
dataset.apply(stats.zscore)

stats.zscore(dataset['Salary'])

#degree of freedom 
a=dataset.shape[0]h
b=dataset.shape[1]

y_mean=np.mean(y)
SSR=np.sum((y_pred-y_mean)**2)
print(SSR)

from sklearn.metrics import mean_squared_error

filename = 'linear_regression_model.pkl'
with open(regressor, 'wb') as file:
    pickle.dump(regressor, file)
print("Model has been pickled and saved as linear")

dataset.skew()

    

g

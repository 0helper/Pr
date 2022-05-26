import numpy as np
import matplotlib.pyplot as plt 

import pandas as pd  
import seaborn as sns 
from sklearn.datasets import load_boston
boston_dataset = load_boston()
df=pd.read_csv("test_data.csv")
print(df)
print(boston_dataset.keys())
print(boston_dataset.DESCR)

df_boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
print(df_boston.head())

df_boston['MEDV'] = boston_dataset.target

print(df_boston.isnull().sum())

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(df_boston['MEDV'], bins=30)
plt.show()

correlation_matrix = df_boston.corr().round(2)

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()

plt.figure(figsize=(20, 5))

features = ['LSTAT', 'RM']
target = df_boston['MEDV']

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = df_boston[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')

    plt.show()

    X = pd.DataFrame(np.c_[df_boston['LSTAT'], df_boston['RM']], columns = ['LSTAT','RM'])
Y = df_boston['MEDV']


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)


from sklearn.metrics import r2_score

y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")


y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))





import numpy as np
import pandas as pd

data_frame=pd.read_csv("Iris.csv")
print(data_frame)

print(data_frame.head())

print(data_frame.tail())

print(data_frame.describe())
print(data_frame.info())
print(data_frame.shape)
print(data_frame.isnull().any().any())
print(data_frame.isnull().sum())
print(data_frame.dtypes)
avg_val=data_frame["petal.length"].astype("float").mean()
data_frame["petal.length"].replace(np.NaN,avg_val, inplace=True)
print(data_frame["petal.length"])

print(data_frame.dtypes)
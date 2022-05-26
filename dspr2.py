import numpy as np
import pandas as pd

data_frame=pd.read_csv("D:\ROLL_NO_28_DSBDA\Academic_performace.csv")
print(data_frame)

print(data_frame.head())
print(data_frame.tail())
print(data_frame.describe())
print(data_frame.info())
print(data_frame.shape)
print(data_frame.isnull().any().any())
print(data_frame.isnull().sum())

avg_val=data_frame["Discussion"].astype("float").mean()
print(avg_val)

data_frame["Discussion"].replace(np.NaN,avg_val, inplace=True)
print(data_frame["Discussion"])


import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
x = data_frame['Sno']
y = data_frame['AnnouncementsView']
sns.regplot(x, y)

sns.boxplot(x=data_frame['AnnouncementsView'])
z = np.abs(stats.zscore(data_frame['AnnouncementsView']))
print(z)
threshold = 3
print(np.where(z > 3))

df = pd.DataFrame({
'Income': [15000, 1800, 120000, 10000],
'Age': [25, 18, 42, 51],
'Department': ['HR','Legal','Marketing','Management']
})
print(df)
df_scaled = df.copy()
col_names = ['Income', 'Age']
features = df_scaled[col_names]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled[col_names] = scaler.fit_transform(features.values)
print(df_scaled[col_names])
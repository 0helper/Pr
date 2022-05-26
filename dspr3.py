import numpy as np
import pandas as pd
import statistics as st 

df = pd.read_csv("test_AV3.csv")
print(df.info())

print(df.shape) 
print(df.info())
print(df.mean())


print(df.loc[:,'CoapplicantIncome'].mean())
print(df.loc[:,'ApplicantIncome'].mean()) 

df.mean(axis = 1)[0:2]
print(df.median())
print(df.loc[:,'ApplicantIncome'].median())
print(df.loc[:,'CoapplicantIncome'].median())

print(df.mode())
print(df.std())

print(df.loc[:,'ApplicantIncome'].std())
print(df.loc[:,'CoapplicantIncome'].std())

print(df.var())

from scipy.stats import iqr
print(iqr(df['ApplicantIncome']))

print(df.skew())
print(df.describe())
print(df.describe(include='all'))




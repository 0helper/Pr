import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("D:\ROLL_NO_28_DSBDA/titanic.csv")
dataset.head()
print(dataset.head())
sns.histplot(dataset['Fare'], kde=False, bins=10)
#print(sns.histplot)
plt.show()

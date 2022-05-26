import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("titanic.csv")
dataset.head()
print(dataset.head())
sns.histplot(dataset['Fare'], kde=False, bins=10)

#print(sns.histplot)
plt.show()
g=sns.boxplot(x='Sex', y='Age', data=dataset, hue="Survived")
plt.show()
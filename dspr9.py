import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("titanic.csv")


g=sns.boxplot(x='Sex', y='Age', data=dataset, hue="Survived")
plt.show()
print(dataset.head())

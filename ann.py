# Importing the libraries
import numpy as np
import matplotlib as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
##### DATA PREPROCESSING #####

# Importing the libraries
import numpy as np
import matplotlib as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13].values

# Encoding Categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
ct = ColumnTransformer(
        [('oh_enc', OneHotEncoder(sparse=False), [1, 2]),],
        remainder='passthrough'
    )

X = ct.fit_transform(X)

# Avoid dummy variables trap
X = np.delete(X, [0, 3], axis=1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
X_train = standardScaler.fit_transform(X_train)
X_test = standardScaler.transform(X_test)


##### Create ANN #####
import keras
from keras.models import Sequential
from keras.layers import Dense

# Init ANN
classifier = Sequential()

# Adding the Input Layer and the first Hidden Layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second Hiddem Layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the Output Layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
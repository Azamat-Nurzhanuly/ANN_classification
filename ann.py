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
X = np.delete(X, [0, 3], axis = 1)

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

# Compiling the  ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

##### Prediction and Evaluation #####

# Making prediction
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Example of single new observation
"""
Predict if the customer with the following information will leave the bank:
    Geography: France
    Credit Score: 600
    Gender: Male
    Age: 40 years old
    Tenure: 3 years
    Balance: $60000
    Number of Products: 2
    Does this customer have a credit card ? Yes
    Is this customer an Active Member: Yes
    Estimated Salary: $50000
"""
x = np.array([[600, 'France', 'Male', 40, 3, 6000, 2, 1, 1, 50000]])
x = ct.transform(x)
x = np.delete(x, [0, 3], axis = 1)
x = standardScaler.transform(x)
y_predict = classifier.predict(x)
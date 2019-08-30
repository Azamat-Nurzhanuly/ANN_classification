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
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# Init ANN
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)

# Cross validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
mean = accuracies.mean()
variance = accuracies.std()

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
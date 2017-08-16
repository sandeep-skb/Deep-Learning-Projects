#Importing library
import numpy as np # To work with arrays
import matplotlib.pyplot as plt # for plots
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

# Matrix of IV
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder();
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder();
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])


onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]




# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting classifier to the Training set
# Create your classifier here
# Make ANN
# Importing keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(units=6, activation="relu", input_dim=11, kernel_initializer="uniform"))

#Adding the second hidden layer
classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform")) #kernel_initializer initializes 
# weights randomly and close to 0

# Adding the output layer
classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))

# compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



# Read the test set 
testset = pd.read_csv('Homework_test.csv')
# extracting IV which are of importance
x_test = testset.iloc[:, 3:13].values

#transforming string IVs
x_test[:, 1] = labelencoder_X_1.transform(x_test[:, 1])
x_test[:, 2] = labelencoder_X_2.transform(x_test[:, 2])

#encoding transformed string IVs
x_test = onehotencoder.transform(x_test).toarray()
x_test = x_test[:, 1:]

# Feature Scaling on fitted X
x_test = sc_X.transform(x_test)

# Predict 
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)






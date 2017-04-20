# Load libraries
import pandas as pd #Data analysis library
from pandas.tools.plotting import scatter_matrix #Graphics library
import matplotlib.pyplot as plt #Graphics library
from sklearn import model_selection #Machine learning models
from sklearn.metrics import classification_report #Build a text report showing the main classification metrics
from sklearn.metrics import confusion_matrix #Compute confusion matrix to evaluate the accuracy of a classification
from sklearn.metrics import accuracy_score #In multilabel classification, this function computes subset accurac
from sklearn.linear_model import LogisticRegression #Charge LogisticRegression model
from sklearn.tree import DecisionTreeClassifier #Charge DecisionTreeClassifier model
from sklearn.neighbors import KNeighborsClassifier #Charge KNeighborsClassifier model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #Charge LinearDiscriminantAnalysis model
from sklearn.naive_bayes import GaussianNB #Charge GaussianNB model
from sklearn.svm import SVC #Used to load the data
import io
import requests

# Load dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] #Parameters names of the data
dataset = pd.read_csv("IRIS.csv", names=names) #Preparing dataset associate data with out parameters names

# Split-out validation dataset
array = dataset.values #Insert the dataset into an array
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.80 #Size of the training dataset
seed = 7 #random seed to test where the data is going to start
scoring = 'accuracy' #the precission
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Spot Check Algorithms
models = [] #ML models
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed) #Split dataset into k consecutive folds
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring) #collects all the results
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

knn = KNeighborsClassifier() #KNeighborsClassifier was the one with best accuracy_score
knn.fit(X_train, Y_train) #fit the data to train knn
predictions = knn.predict(X_validation) #get the predictiones from the training
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

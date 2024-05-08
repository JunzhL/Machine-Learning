def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

path = './mushroom.csv'
df = pd.read_csv(path)
print(df.head())

# Data Preprocessing
print(df.isnull().sum())
print(df.dtypes)

X = df.drop('class', axis=1)
y = df['class']

# Encode the categorical values
le = preprocessing.LabelEncoder()
X = X.apply(le.fit_transform)
y = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
lg = LogisticRegression()
lg.fit(X_train, y_train)
lg_pred = lg.predict(X_test)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# SVM
svm = svm.SVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

# Plot as bar graph for each model's accuracy
models = ['Logistic Regression', 'KNN', 'Decision Tree', 'SVM']
accuracy = [accuracy_score(y_test, lg_pred), accuracy_score(y_test, knn_pred), accuracy_score(y_test, dt_pred), accuracy_score(y_test, svm_pred)]
plt.bar(models, accuracy)
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.show()





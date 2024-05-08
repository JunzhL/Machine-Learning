# Surpress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
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

path = './Weather_Data.csv'
df = pd.read_csv(path)
print(df.head())

# Data Preprocessing
df_processed = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
df_processed.replace(['No', 'Yes'], [0,1], inplace=True)
df_processed.drop('Date',axis=1,inplace=True)
df_processed = df_processed.astype(float)
features = df_processed.drop('RainTomorrow', axis=1)
target = df_processed['RainTomorrow']

print("Linear Regression:")
# Linear Regression
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=4)
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# the accuracy of the model
mae = metrics.mean_absolute_error(y_test, lr_pred)
print('Mean Absolute Error:', mae)
mse = metrics.mean_squared_error(y_test, lr_pred)
print('Mean Squared Error:', mse)
r2 = metrics.r2_score(y_test, lr_pred)
print('R2 Score:', r2)

lr_report = pd.DataFrame(data=[[mae, mse, r2]], columns=["MAE", "MSE", "R2"])
print(lr_report)

print("Logistic Regression:")
# Logistic Regression
logr = LogisticRegression()
logr.fit(X_train, y_train)
log_pred = logr.predict(X_test)

# the accuracy of the model
accuracy = accuracy_score(y_test, log_pred)
print('Accuracy:', accuracy)
jaccard = jaccard_score(y_test, log_pred)
print('Jaccard Score:', jaccard)
f1 = f1_score(y_test, log_pred)
print('F1 Score:', f1)
logloss = log_loss(y_test, log_pred)
print('Log Loss:', logloss)

logr_report = pd.DataFrame(data=[[accuracy, jaccard, f1, logloss]], columns=["Accuracy", "Jaccard", "F1", "Log Loss"])
print(logr_report)

print("KNN Classifier:")
# KNN Classifier
KNN = KNeighborsClassifier(n_neighbors=5)
KNN.fit(X_train, y_train)
KNN_pred = KNN.predict(X_test)

# the accuracy of the model
accuracy = accuracy_score(y_test, KNN_pred)
print('Accuracy:', accuracy)
jaccard = jaccard_score(y_test, KNN_pred)
print('Jaccard Score:', jaccard)
f1 = f1_score(y_test, KNN_pred)
print('F1 Score:', f1)

KNN_report = pd.DataFrame(data=[[accuracy, jaccard, f1]], columns=["Accuracy", "Jaccard", "F1"])
print(KNN_report)

# Decision Tree
print("Decision Tree:")
dt = DecisionTreeClassifier(criterion="entropy", max_depth=6)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# the accuracy of the model
accuracy = accuracy_score(y_test, dt_pred)
print('Accuracy:', accuracy)
jaccard = jaccard_score(y_test, dt_pred)
print('Jaccard Score:', jaccard)
f1 = f1_score(y_test, dt_pred)
print('F1 Score:', f1)

dt_report = pd.DataFrame(data=[[accuracy, jaccard, f1]], columns=["Accuracy", "Jaccard", "F1"])
print(dt_report)

# compare the models by plotting the accuracy, jaccard, and f1 scores
fig, ax = plt.subplots(figsize=(10, 10))
barWidth = 0.25
bars1 = [lr_report['MAE'][0], logr_report['Accuracy'][0], KNN_report['Accuracy'][0], dt_report['Accuracy'][0]]
bars2 = [lr_report['R2'][0], logr_report['Jaccard'][0], KNN_report['Jaccard'][0], dt_report['Jaccard'][0]]
bars3 = [0, logr_report['F1'][0], KNN_report['F1'][0], dt_report['F1'][0]]
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

plt.bar(r1, bars1, color='b', width=barWidth, edgecolor='grey', label='Accuracy')
plt.bar(r2, bars2, color='r', width=barWidth, edgecolor='grey', label='Jaccard')
plt.bar(r3, bars3, color='g', width=barWidth, edgecolor='grey', label='F1')

plt.xlabel('Models', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], ['Linear Regression', 'Logistic Regression', 'KNN Classifier', 'Decision Tree'])
plt.legend()
plt.show()



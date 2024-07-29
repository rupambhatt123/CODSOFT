from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import mean_absolute_errimport numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.svm import SVC
import sklearn.metrics as metrics
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_lossor
from sklearn.metrics import mean_squared_error


df = pd.read_csv("/kaggle/input/fraud-detection/fraudTrain.csv")
df

df.isnull().sum()

df.dtypes

df.drop(columns=["Unnamed: 0", "trans_num", "street"], inplace= True)
df

data = df.head(n = 20000)
data.is_fraud.value_counts()

df_processed = pd.get_dummies(data=data)
df_processed

x_train = df_processed.drop(columns='is_fraud', axis=1)
y_train = df_processed['is_fraud']


df_test = pd.read_csv("/kaggle/input/fraud-detection/fraudTest.csv")
df_test

df_test.drop(columns=["Unnamed: 0", "trans_num", "street"], inplace= True)
df_test

data_test = df_test.sample(frac=1, random_state=1).reset_index()
data_test = data_test.head(n = 5000)
data_test.is_fraud.value_counts()

df_processed_test = pd.get_dummies(data=data_test)
df_processed_test

x_test = df_processed.drop(columns='is_fraud', axis=1)
y_test = df_processed['is_fraud']


LR = LogisticRegression(solver='liblinear')

LR.fit(x_train, y_train)

predictions = LR.predict(x_test)

predict_proba = LR.predict_proba(x_test)

LR_Accuracy_Score = accuracy_score(y_test, predictions)

print(LR_Accuracy_Score)


Tree = DecisionTreeClassifier()


Tree.fit(x_train, y_train)

predictions = Tree.predict(x_test)

Tree_Accuracy_Score = accuracy_score(y_test, predictions)
Tree_JaccardIndex = jaccard_score(y_test, predictions)
Tree_F1_Score = f1_score(y_test, predictions)


print(Tree_Accuracy_Score)
print(Tree_JaccardIndex)
print(Tree_F1_Score)


knn = KNeighborsClassifier(n_neighbors=4) 
knn.fit(x_train, y_train)

predictions = knn.predict(x_test)

KNN_Accuracy_Score = accuracy_score(y_test, predictions)
KNN_JaccardIndex = jaccard_score(y_test, predictions)
KNN_F1_Score = f1_score(y_test, predictions)

print(KNN_Accuracy_Score)
print(KNN_JaccardIndex)
print(KNN_F1_Score)
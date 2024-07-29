import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler

churn_df = pd.read_csv("/kaggle/input/bank-customer-churn-prediction/Churn_Modelling.csv")

churn_df = churn_df.copy()
churn_df.head()

churn_df.info()

churn_df.describe()

churn_df["CustomerId"].nunique()

null_values = churn_df.isnull().sum().sum()
print(f"Number of Null values : {null_values}")

churn_df.drop(columns=["RowNumber","CustomerId","Surname"],inplace = True)

list(churn_df.columns)

churn_df.head()

churn_df["Tenure"].value_counts().values

plt.figure(figsize=(10, 6))
sns.histplot(churn_df['CreditScore'], bins=30, kde=True, color='royalblue')
plt.title('Distribution of Credit Scores')
plt.xlabel('Credit Score')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=churn_df, x='Geography', palette='Set2')
plt.title('Geographical Distribution')
plt.xlabel('Geography')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=churn_df, x='Gender', palette='coolwarm')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

gender_counts = churn_df['Gender'].value_counts()
labels = gender_counts.index
sizes = gender_counts.values

colors = sns.color_palette('coolwarm', len(labels))

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Gender Distribution')
plt.axis('equal')  

plt.figure(figsize=(10,6))
sns.histplot(data = churn_df,x="Age",kde = True,bins = 30,color = "darkorange")
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(churn_df['Tenure'], bins=10, kde=True, color='teal')
plt.title('Tenure Distribution')
plt.xlabel('Tenure (Years)')
plt.ylabel('Frequency')
plt.show()

tenure = churn_df["Tenure"].value_counts()
plt.figure(figsize = (12,6))
sns.countplot(data=churn_df,x="Tenure",palette = "viridis")
plt.title("Tenure Distribution")
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(churn_df['Balance'], bins=35, kde=True, color='purple')
plt.title('Balance Distribution')
plt.xlabel('Balance')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=churn_df, x='NumOfProducts', palette='Set1')
plt.title('Number of Products')
plt.xlabel('Number of Products')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=churn_df, x='HasCrCard', palette='Set2')
plt.title('Credit Card Holders')
plt.xlabel('Has Credit Card')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(churn_df['EstimatedSalary'], bins=30, kde=True, color='coral')
plt.title('Estimated Salary Distribution')
plt.xlabel('Estimated Salary')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(data=churn_df,x='Exited',y='Age')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=churn_df, x='Exited', palette='Set3')
plt.title('Churn Distribution')
plt.xlabel('Exited')
plt.ylabel('Count')
plt.show()

churn_counts = churn_df['Exited'].value_counts()
labels = churn_counts.index
sizes = churn_counts.values

colors = sns.color_palette('Set3', len(labels))

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Churn Distribution')
plt.axis('equal')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=churn_df, x='CreditScore', y='EstimatedSalary', hue='Exited', palette='plasma')
plt.title('Credit Score vs. Estimated Salary')
plt.xlabel('Credit Score')
plt.ylabel('Estimated Salary')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=churn_df, x='Age', y='Balance', hue='Exited', palette='viridis')
plt.title('Age vs. Balance')
plt.xlabel('Age')
plt.ylabel('Balance')
plt.show()

churn_encoded_df = pd.get_dummies(churn_df, columns=['Geography', 'Gender'], drop_first=True)

plt.figure(figsize=(12, 10))
sns.heatmap(churn_encoded_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

churn_df = pd.get_dummies(churn_df, columns=['Geography'],dtype=int,drop_first=True)

input_col = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Geography_Germany','Geography_Spain']
target_col = "Exited"
input_col

churn_df["Gender"] = churn_df["Gender"].map({"Male": 1, "Female": 0}) 

churn_train_df, churn_test = train_test_split(churn_df,test_size = 0.2,random_state=42)
churn_train , churn_val = train_test_split(churn_train_df,test_size=0.25,random_state=42)
len(churn_train) , len(churn_test) , len(churn_val)

churn_df.info()

train_input = churn_train[input_col].copy()
train_target = churn_train[target_col].copy()
test_input = churn_test[input_col].copy()
test_target = churn_test[target_col].copy()
val_input = churn_val[input_col].copy()
val_target = churn_val[target_col].copy()

train_input.info()

list(churn_df.columns)

numeric_cols = ['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Gender']
categorical_col = ['Geography_Germany','Geography_Spain']

scaler = StandardScaler()
scaler.fit(train_input[numeric_cols])

train_input[numeric_cols] = scaler.transform(train_input[numeric_cols])
test_input[numeric_cols] = scaler.transform(test_input[numeric_cols])

val_input[numeric_cols] = scaler.transform(val_input[numeric_cols])

X_train = train_input[numeric_cols+categorical_col]
X_val = val_input[numeric_cols+categorical_col]
X_test = test_input[numeric_cols+categorical_col]

model_DTC = DecisionTreeClassifier(random_state = 42,max_depth=7,max_leaf_nodes=30).fit(X_train,train_target)
train_preds_dtc = model_DTC.score(X_train,train_target)
val_preds_dtc = model_DTC.score(X_val,val_target)
test_preds_dtc = model_DTC.score(X_test,test_target)
print(f"Train Accuracy : {train_preds_dtc}")
print(f"Val Accuracy : {val_preds_dtc}")
print(f"Test Accuracy : {test_preds_dtc}")

model_rfc = RandomForestClassifier(n_jobs=-1,random_state=42,max_depth=10,max_leaf_nodes= 80,n_estimators = 30).fit(X_train,train_target)
train_preds_rfc = model_rfc.score(X_train,train_target)
val_preds_rfc = model_rfc.score(X_val,val_target)
test_preds_rfc = model_rfc.score(X_test,test_target)
print(f"Train Accuracy : {train_preds_rfc}")
print(f"Val Accuracy : {val_preds_rfc}")
print(f"Test Accuracy : {test_preds_rfc}")

model_xgbc = XGBClassifier(n_jobs=-1,random_state=42).fit(X_train,train_target)
train_preds_xgbc = model_xgbc.score(X_train,train_target)
val_preds_xgbc = model_xgbc.score(X_val,val_target)
test_preds_xgbc = model_xgbc.score(X_test,test_target)
print(f"Train Accuracy : {train_preds_xgbc}")
print(f"Val Accuracy : {val_preds_xgbc}")
print(f"Test Accuracy : {test_preds_xgbc}")
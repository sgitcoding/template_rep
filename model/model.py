# import libraries

# 1. to handle the data
import pandas as pd
import numpy as np

# 2. To Viusalize the data
import matplotlib.pyplot as plt
import seaborn as sns

#3.To split the data 
from sklearn.model_selection import train_test_split

#to train the model
#import xgboost as xgb


#to test the model 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score , recall_score , accuracy_score , f1_score

from sklearn.model_selection import GridSearchCV

#loading the data 

df = pd.read_csv('heart_disease_health_indicators_BRFSS2015.csv')

#print the first 5 data samples 

df.head()

#print the last 5 data samples 

df.tail()

df.isnull()

df.isnull().sum()

df.dtypes

#print complete dataframe information

df.info

df.shape

if (df['HeartDiseaseorAttack'] == 0).all():
    print("All values in the column 'HeartDiseaseorAttack' are Zero")
else:
    print("All values in the column 'HeartDiseaseorAttack' are not Zero")

if (df['HeartDiseaseorAttack'] == 0).any():
    print("Some values in the column 'HeartDiseaseorAttack' are Zero")
else:
    print("Some values in the column 'HeartDiseaseorAttack' are Non Zero")

df.describe()


sns.histplot(df['Age'])

df.columns
col = ['HighBP', 'HighChol', 'CholCheck',
       'Smoker', 'Stroke', 'Diabetes', 'PhysActivity', 'Fruits', 'Veggies',
       'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
       'DiffWalk', 'Sex', 'Education',
       'Income']
plt.figure(figsize=(5,30))
for i,column in enumerate(col):
    plt.subplot(len(col), 2, i+1)
    plt.suptitle("Countplot of Categories", fontsize=15, x=0.5, y=1)
    sns.countplot(data=df, x=column)
    plt.title(f"{column}")
    plt.tight_layout()

#correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=True, cmap='Purples')
plt.title('Correlation Matrix')
plt.show()


hd = (df["HeartDiseaseorAttack"] == 1.0).sum()
nohd = (df["HeartDiseaseorAttack"] == 0.0).sum()

sizes = [hd, nohd]
labels = ['Heart Disease', 'No Heart Disease']

plt.figure(figsize=(6, 4))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Heart Disease Rate')

plt.axis('equal')
plt.show()

x = df.drop('HeartDiseaseorAttack', axis =1)
y = df['HeartDiseaseorAttack']

#splitting the data for training and testing the data 

x_train, x_test , y_train, y_test=train_test_split(x, y, random_state=200, test_size=0.25, shuffle= True )

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
#from xgboost import XGBClassifier  
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

%%time

dt=DecisionTreeClassifier()
dt.fit(x_train, y_train)

print("train shape: " + str(x_train.shape))
print("score on test: " + str(dt.score(x_test, y_test)))
print("score on train: "+ str(dt.score(x_train, y_train)))

dt_y_predicts = dt.predict(x_test)

print(dt_y_predicts[:15])
print(y_test[:15])

accuracy = accuracy_score(y_test, dt_y_predicts)
print('Accuracy = '+ str(accuracy))

f1 = f1_score(y_test, dt_y_predicts)
print('f1 score = '+ str(f1))

recall = recall_score(y_test, dt_y_predicts)
print('recall = '+ str(recall))

precision = precision_score(y_test, dt_y_predicts)
print('precision = '+ str(precision))

dt_cm = confusion_matrix(y_test, dt_y_predicts)
print('confusion matrix'+ str(dt_cm))


%%time

lr=LogisticRegression(max_iter=1000)
lr.fit(x_train, y_train)

print("train shape: " + str(x_train.shape))
print("score on test: " + str(lr.score(x_test, y_test)))
print("score on train: "+ str(lr.score(x_train, y_train)))

lr_y_predicts = lr.predict(x_test)

print(lr_y_predicts[:15])
print(y_test[:15])

accuracy = accuracy_score(y_test, lr_y_predicts)
print('Accuracy = '+ str(accuracy))

f1 = f1_score(y_test, lr_y_predicts)
print('f1 score = '+ str(f1))

recall = recall_score(y_test, lr_y_predicts)
print('recall = '+ str(recall))

precision = precision_score(y_test, lr_y_predicts)
print('precision = '+ str(precision))

lr_cm = confusion_matrix(y_test, lr_y_predicts)
print('confusion matrix'+ str(lr_cm))



%%time

rfc= RandomForestClassifier(n_estimators=100 , max_depth=9)
rfc.fit(x_train, y_train)

print("train shape: " + str(x_train.shape))
print("test accuracy: " + str(rfc.score(x_test,y_test)))
print("train accuracy: " + str(rfc.score(x_train,y_train)))

rfc_y_predicts = rfc.predict(x_test)

print(rfc_y_predicts[:15])
print(y_test[:15])

accuracy = accuracy_score(y_test, rfc_y_predicts)
print('Accuracy = '+ str(accuracy))

f1 = f1_score(y_test, rfc_y_predicts)
print('f1 score = '+ str(f1))

recall = recall_score(y_test, rfc_y_predicts)
print('recall = '+ str(recall))

precision = precision_score(y_test, rfc_y_predicts)
print('precision = '+ str(precision))

rfc_cm = confusion_matrix(y_test, rfc_y_predicts)
print('confusion matrix'+ str(rfc_cm))


#with hyperparameter tuning
%%time

rfc= RandomForestClassifier(n_estimators=100 , max_depth=9)
rfc.fit(x_train, y_train)

print("train shape: " + str(x_train.shape))
print("test accuracy: " + str(rfc.score(x_test,y_test)))
print("train accuracy: " + str(rfc.score(x_train,y_train)))

rfc_y_predicts = rfc.predict(x_test)

print(rfc_y_predicts[:15])
print(y_test[:15])

accuracy = accuracy_score(y_test, rfc_y_predicts)
print('Accuracy = '+ str(accuracy))

f1 = f1_score(y_test, rfc_y_predicts)
print('f1 score = '+ str(f1))

recall = recall_score(y_test, rfc_y_predicts)
print('recall = '+ str(recall))

precision = precision_score(y_test, rfc_y_predicts)
print('precision = '+ str(precision))

rfc_cm = confusion_matrix(y_test, rfc_y_predicts)
print('confusion matrix'+ str(rfc_cm))

%%time

from sklearn.naive_bayes import BernoulliNB

bnb= BernoulliNB()
bnb.fit(x_train, y_train)

print("train shape: " + str(x_train.shape))
print("test accuracy: " + str(bnb.score(x_test,y_test)))
print("train accuracy: " + str(bnb.score(x_train,y_train)))

bnb_y_predicts = bnb.predict(x_test)

print(bnb_y_predicts[:15])
print(y_test[:15])

accuracy = accuracy_score(y_test, bnb_y_predicts)
print('Accuracy = '+ str(accuracy))

f1 = f1_score(y_test, bnb_y_predicts)
print('f1 score = '+ str(f1))

recall = recall_score(y_test, bnb_y_predicts)
print('recall = '+ str(recall))

precision = precision_score(y_test,bnb_y_predicts)
print('precision = '+ str(precision))

bnb_cm = confusion_matrix(y_test, bnb_y_predicts)
print('confusion matrix'+ str(bnb_cm))


%%time

gbc= GradientBoostingClassifier(n_estimators=100)
gbc.fit(x_train, y_train)

print("train shape: " + str(x_train.shape))
print("test accuracy: " + str(gbc.score(x_test,y_test)))
print("train accuracy: " + str(gbc.score(x_train,y_train)))

gbc_y_predicts = gbc.predict(x_test)

print(gbc_y_predicts[:15])
print(y_test[:15])

accuracy = accuracy_score(y_test, gbc_y_predicts)
print('Accuracy = '+ str(accuracy))

f1 = f1_score(y_test, gbc_y_predicts)
print('f1 score = '+ str(f1))

recall = recall_score(y_test, gbc_y_predicts)
print('recall = '+ str(recall))

precision = precision_score(y_test, gbc_y_predicts)
print('precision = '+ str(precision))

gbc_cm = confusion_matrix(y_test, gbc_y_predicts)
print('confusion matrix'+ str(gbc_cm))




'''confusion_matrix
accuracy_score
recall_score
precision_score
f1_score
roc_curve
roc_auc_score'''




#dataset taken from kaggle 

# import libraries

# 1. to handle the data
import pandas as pd
import numpy as np

# 2. To Viusalize the data
import matplotlib.pyplot as plt
import seaborn as sns

#3.To split the data 
from sklearn.model_selection import train_test_split

import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense

import tensorflow as tf



import keras_tuner 
from keras_tuner.tuners import RandomSearch

#to test the model 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score , recall_score , accuracy_score , f1_score

df = pd.read_csv('heart_disease_health_indicators_BRFSS2015.csv')

#print the first 5 data samples 

df.head()

#print the last 5 data samples 

df.tail()

df.isnull().sum()

df.dtypes

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

x_train, x_test , y_train, y_test=train_test_split(x, y, random_state=42 , test_size=0.25, shuffle= True )

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)



#hyperparameter tuning using keras_tuner 
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Dense(
        units=hp.Choice('units', [32,64,128,256,512,600,700,800, 900]),
        activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

tuner_search = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    directory='output_6',
    project_name="heart_disease_prediction_6"
)

tuner_search.search(x_train, y_train, epochs=2, validation_data=(x_test, y_test))

best_models = tuner_search.get_best_models(num_models=1)

test_loss, test_acc = best_models[0].evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

predictions = best_models[0].predict(x_test)
# For binary classification, use a threshold of 0.5
predicted_classes = (predictions > 0.5).astype(int).flatten()


accuracy = accuracy_score(y_test,predicted_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test,predicted_classes,average='weighted')
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test,predicted_classes,average='weighted')
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test,predicted_classes,average='weighted')
print('F1 score: %f' % f1)



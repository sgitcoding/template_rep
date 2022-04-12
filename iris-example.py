from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import shap
import os
import joblib
import dill
import numpy as np
import boto3
import dill
from io import BytesIO
import json

# GLOBALS
SEED = 42
model_dir = "./model"
explainer_dir = "./explainer"
MODEL_FILE = "model.joblib"
EXPLAINER_FILE = "explainer.dill"
use_reference = False
bucket_model = 'deeploy-examples'
bucket_explainer = 'deeploy-examples'
explainer_object_key = 'sklearn/iris/explainer/explainer.dill'
model_object_key = 'sklearn/iris/model/model.joblib'

# Load Iris data
iris = load_iris()
y = iris['target']
X = iris['data']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# Fit RF
rf = RandomForestClassifier(
    criterion="entropy",
    n_estimators=20,
    min_samples_split=10,
    max_depth=10,
    min_samples_leaf=1,
    max_features="auto",
    random_state=1,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

# Fit Shap Kernel Explainer
f = lambda x: rf.predict_proba(x)[:,1]
explainer = shap.KernelExplainer(f, X_train) # with large X_train in stead use median

# Export model file
model_file = os.path.join((model_dir), MODEL_FILE)
joblib.dump(rf, model_file, compress=3)

# Export explainer file
explainer_file = os.path.join((explainer_dir), EXPLAINER_FILE)
with open(f'{explainer_file}', 'wb') as handle:
    dill.dump(explainer, handle)

if use_reference:
    # Export model objects
    s3 = boto3.resource('s3')
    s3.meta.client.upload_file('model/model.joblib', bucket_model, model_object_key)
    s3.meta.client.upload_file('explainer/explainer.dill', bucket_explainer, explainer_object_key)

    # Add reference to repo
    model_reference = {
        'reference': {
            'blob': {
                'url': 's3://' + bucket_model + '/' + model_object_key.strip(MODEL_FILE)
                }
            }
        }
    explainer_reference = {
        'reference': {
            'blob': {
                'url': 's3://' + bucket_explainer + '/' + explainer_object_key.strip(EXPLAINER_FILE)
                }
            }
        }
    
    with open('model/reference.json', 'w', encoding='utf-8') as f:
        json.dump(model_reference, f, ensure_ascii=False, indent=4)
    with open('explainer/reference.json', 'w', encoding='utf-8') as f:
        json.dump(explainer_reference, f, ensure_ascii=False, indent=4)

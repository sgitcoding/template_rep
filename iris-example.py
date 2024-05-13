import xgboost as xgb
from sklearn.datasets import load_iris
import os
import dill
from shap import KernelExplainer
import numpy as np


model_dir = "./model"
BST_FILE = "model.bst"

exp_dir = "./explainer"
SHAP_FILE = "explainer.dill"

iris = load_iris()
y = iris['target']
X = iris['data']
xgbclass = xgb.XGBClassifier(max_depth=6)
xgbclass.fit(X,y)
model_file = os.path.join((model_dir), BST_FILE)
xgbclass.save_model(model_file)

predict_fn = lambda x: xgbclass.predict(np.array(x))
explainer = KernelExplainer(predict_fn, X[:100])
explainer.model = None
with open(os.path.join(exp_dir, SHAP_FILE), "wb") as f:
    dill.dump(explainer, f)

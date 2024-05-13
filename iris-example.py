import xgboost as xgb
from sklearn.datasets import load_iris
import os
import dill
from shap import KernelExplainer


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

explainer = KernelExplainer(xgbclass.predict, X[:100])
explainer.model = None
with open(os.path.join(exp_dir, SHAP_FILE), "wb") as f:
    dill.dump(explainer, f)

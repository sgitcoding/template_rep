import xgboost as xgb
from sklearn.datasets import load_iris
import os
import dill
from shap import KernelExplainer
import numpy as np


class KernelExplainer(KernelExplainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def shap_values(self, X, **kwargs):
        vals = super().shap_values(X, **kwargs)
        return [val.tolist() for val in vals]


model_dir = "./model"
BST_FILE = "model.bst"

exp_dir = "./explainer"
SHAP_FILE = "explainer.dill"

iris = load_iris()
y = iris['target']
X = iris['data']
model = xgb.Booster()

# Train xgboost model
dtrain = xgb.DMatrix(X, label=y)
params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "max_depth": 3
}
model = xgb.train(params, dtrain, num_boost_round=10)

model_file = os.path.join((model_dir), BST_FILE)
model.save_model(model_file)

predict_fn = lambda x: model.predict(xgb.DMatrix(np.array(x)))
explainer = KernelExplainer(predict_fn, X[:100])
explainer.model = None
with open(os.path.join(exp_dir, SHAP_FILE), "wb") as f:
    dill.dump(explainer, f)

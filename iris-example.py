import xgboost as xgb
from sklearn.datasets import load_iris
import os

model_dir = "./model"
BST_FILE = "model.bst"

iris = load_iris()
y = iris['target']
X = iris['data']
xgbclass = xgb.XGBClassifier(max_depth=6)
xgbclass.fit(X,y)
model_file = os.path.join((model_dir), BST_FILE)
xgbclass.save_model(model_file)

import sys

sys.path.append("../1 - Decision Tree Regressor")

from DecisionTreeRegressor import DecisionTreeRegressor
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_percentage_error

class GradientBoostingRegressor():

    def __init__(self, max_depth=2,n_estimators=50):
        self.max_tree_depth = max_depth
        self.number_of_estimators = n_estimators
        self.trees = []
        self.learning_rate = None
        self.initial_model = None

    def fit(self,x,y,learning_rate=0.01):
        model = self._initialize_model(y.values)

        self.learning_rate = learning_rate
        self.initial_model = model

        preds = np.full((x.shape[0],),model)
        for e in range(self.number_of_estimators):
            print(f"Estimator {e+1} is being trained...")
            res = y.values.reshape((y.shape[0],)) - preds

            res = pd.DataFrame(res,index=y.index)
            decision_tree = DecisionTreeRegressor(x=x,y=res,max_depth=self.max_tree_depth)
            decision_tree.fit()

            preds += learning_rate * decision_tree.predict(x).reshape((x.shape[0],))
            print(f"Training loss : {mean_absolute_percentage_error(preds,y.values)}")

            self.trees.append(decision_tree)

    def predict(self,x):

        yhat = np.full((x.shape[0],),self.initial_model)
        for tree in self.trees:
            yhat += self.learning_rate*tree.predict(x).reshape((x.shape[0],))

        return yhat

    def _initialize_model(self,y):
        yhat = np.sum(y) / y.shape[0]
        return yhat


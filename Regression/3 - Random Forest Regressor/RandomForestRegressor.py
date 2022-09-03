import sys

sys.path.append("../1 - Decision Tree Regressor")

from DecisionTreeRegressor import *

class RandomForestRegressor():

    def __init__(self, number_of_estimators=10,min_sample=1,max_depth=100,number_of_features=None):
        self.number_of_estimators = number_of_estimators
        self.number_of_features = number_of_features
        self.min_sample = min_sample
        self.max_depth = max_depth
        self.trees = []

    def fit(self,x,y):
        for i in range(self.number_of_estimators):
            tree = DecisionTreeRegressor(x,y,min_sample=self.min_sample,max_depth=self.max_depth,number_of_features=self.number_of_features)
            
            tree.fit()
            
            self.trees.append(tree)

    def predict(self,values):
        all_predictions = []
        for tree in self.trees:
            all_predictions.append(tree.predict(values))

        preds = np.swapaxes(all_predictions,0,1)
        return np.mean(preds,axis=1)

    

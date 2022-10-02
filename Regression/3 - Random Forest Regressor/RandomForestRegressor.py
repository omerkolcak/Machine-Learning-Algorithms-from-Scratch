import sys

sys.path.append("../2 - Decision Tree Regressor")

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
            print(f"Tree {i+1} is being trained...")
            tree = DecisionTreeRegressor(min_sample=self.min_sample,max_depth=self.max_depth,number_of_features=self.number_of_features)
            
            x_bootstraped, y_bootstraped = self.__bootstraping(x,y) 

            tree.fit(x_bootstraped,y_bootstraped)
            
            self.trees.append(tree)

    def __bootstraping(self,x,y):
        number_of_samples = x.shape[0]

        idx = np.random.choice(x.index,number_of_samples,replace=True)

        x = x.loc[idx,:].reset_index(drop=True)
        y = y.loc[idx,:].reset_index(drop=True)

        x.to_csv("x.csv")
        y.to_csv("y.csv")

        return x,y

    def predict(self,values):
        all_predictions = []
        for tree in self.trees:
            all_predictions.append(tree.predict(values))

        preds = np.swapaxes(all_predictions,0,1)
        return np.nanmean(preds,axis=1)

    

from xgboost_tree import XGBoostTree

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

class XGBoostRegressor():
    
    def __init__(self, min_sample=1, max_depth=2, n_estimators=50):
        """
        min_sample:int -> Minimum number of samples on the leaves.
        max_depth:int -> Maximum depth of the trees.
        n_estimator:int -> Number of trees to be trained.
        
        Initialize the model.
        """
        self.min_sample = min_sample
        self.max_tree_depth = max_depth
        self.number_of_estimators = n_estimators
        self.trees = []
        self.learning_rate = None
        self.regularization_lambda = None
        self.gamma = None
        self.initial_model = None

    def fit(self,x,y,learning_rate=0.01,regularization_lambda=0,gamma=0):
        """
        x:pd.DataFrame -> Features of the dataset.
        y:pd.DataFrame -> Target variable.
        learning_rate:float -> Learning rate for the model.
        regularization_lambda:float -> Regularization parameter for the model.
        gamma:float -> Threshold value for pruning the tree. 
        
        This function trains the XGBoost algorithm.
        """
        # initialize base model
        model = self._initialize_base_model(y.values)

        # declare training hyperparameters
        self.learning_rate = learning_rate
        self.regularization_lambda = regularization_lambda
        self.gamma = gamma
        
        self.initial_model = model
        
        preds = np.full((x.shape[0],),model)
        for e in range(self.number_of_estimators):
            print(f"Estimator {e+1} is being trained...")
            res = y.values.reshape((y.shape[0],)) - preds

            residuals = pd.DataFrame(res,index=y.index)
            decision_tree = XGBoostTree(min_sample=self.min_sample,max_depth=self.max_tree_depth
                                        ,gamma=self.gamma,regularization_lambda=self.regularization_lambda)
            
            # fit a decision tree for residuals
            decision_tree.fit(x=x,y=residuals)
            
            # update the prediction with newly trained tree
            preds += learning_rate * decision_tree.predict(x).reshape((x.shape[0],))
            print(f"Training MSE: {mean_squared_error(preds,y.values)}")
            
            # append the trained tree to the list of trees
            self.trees.append(decision_tree)

    def predict(self,x):
        """
        This function performs prediction based on the formula:
        
        F(x) = lr*(Fm(x) + Fm-1(x) + .... F0(x)) 
        
        where Fm(x) denotes the prediction from the mth tree.
        """
        yhat = np.full((x.shape[0],),self.initial_model)
        for tree in self.trees:
            yhat += self.learning_rate*tree.predict(x).reshape((x.shape[0],))

        return yhat

    def _initialize_base_model(self,y):
        """
        This function initializes the base model. Simply return the mean of the target variable, or any arbitrary number. Model will 
        improve this simple prediction.
        """
        return np.mean(y)
import pandas as pd
import numpy as np

def calculate_similarity_score(residuals,regularization_lambda):
    """
    This function calculates the similarity score for given residuals and regularization parameter.
    """
    return (np.sum(residuals)**2).item() / (residuals.shape[0] + regularization_lambda)

class Node():

    def __init__(self,feature=None,threshold=None,left_child=None,right_child=None,values=None,similarity_score=None,is_leaf=False):
        self.feature = feature
        self.threshold = threshold
        self.left = left_child
        self.right = right_child
        self.is_leaf = is_leaf
        self.values = values
        self.similarity_score = similarity_score

class XGBoostTree():
    
    def __init__(self,min_sample=1,max_depth=10,number_of_features=None,gamma=0,regularization_lambda=0):
        self.features = None
        self.number_of_features = number_of_features
        self.root = None
        self.min_sample = min_sample
        self.max_depth = max_depth
        self.gamma = gamma
        self.regularization_lambda = regularization_lambda
        
    def fit(self,x,y):
        """
        x:pd.DataFrame -> Features of the dataset
        y:pd.DataFrame -> Residual values

        This function fits the tree to the given data points.
        """
        self.features = x.columns
        # start to fit the tree
        self.root = self.__expand_tree(x,y,0)

    def __expand_tree(self,x,y,depth):
        """
        x:pd.DataFrame -> Features of the dataset
        y:pd.DataFrame -> Residual values

        This function recursively expands the tree.
        """
        # if you reach the maximum depth or minimum number of samples for a leaf, return the node as a leaf node.
        if depth == self.max_depth or y.shape[0] <= self.min_sample:
            sim_score = calculate_similarity_score(y,self.regularization_lambda)
            return  Node(values=y, similarity_score=sim_score, is_leaf=True) 
        
        # calculate the current similarity socre and pass down to the below branches to calculate the gain
        parent_similarity = calculate_similarity_score(y,self.regularization_lambda)
        left_idx,right_idx,threshold,feature = self.__find_the_attribute(x,y,self.features,parent_similarity)
        
        # if there are no split, then return the node as leaf node.
        if pd.isna(left_idx).any() and pd.isna(right_idx).any():
            sim_score = calculate_similarity_score(y,self.regularization_lambda)
            return Node(values=y, similarity_score=sim_score, is_leaf=True)
        
        # expand the tree to the left 
        left = self.__expand_tree(x.loc[left_idx,:],y.loc[left_idx,:],depth+1)
        # expand the tree to the right
        right = self.__expand_tree(x.loc[right_idx,:],y.loc[right_idx,:],depth+1)
        
        # return the root node for that sub-tree
        return Node(feature=feature,threshold=threshold,left_child=left,right_child=right,values=y,is_leaf=False)

    def __find_the_attribute(self,x,y,features,parent_sim):
        """
        x:pd.DataFrame -> Features of the dataset
        y:pd.DataFrame -> Residual values
        feautures:list(str) -> List of feature names, column names of the x
        parent_sim -> Similarity score of the parent node.
        
        This function loops through all of the features and possible threshold values to find the best split that maximizes the 
        sum of the left child and right chid similarity scores. 
        """
        # declare the best split variables
        max_sim_score = -1000
        best_left_indexes = None
        best_right_indexes = None
        threshold = None
        feature = None

        n = y.shape[0]
        # loop through all features
        for f in features:
            unique_values = list(x[f].unique())
            # loop through all possible threshold values
            for val in unique_values:
                # split by that threshold value
                left_indexes = x[x[f] <= val].index
                right_indexes = x[x[f] > val].index
                left_residuals = y.loc[left_indexes,:]
                right_residuals = y.loc[right_indexes,:]

                # calculate similarity scores for each split
                left_sim_score = calculate_similarity_score(left_residuals,self.regularization_lambda) if left_residuals.shape[0] != 0 else 0
                right_sim_score = calculate_similarity_score(right_residuals,self.regularization_lambda) if right_residuals.shape[0] != 0 else 0
                # and sum them
                sim_score = left_sim_score + right_sim_score

                # if the calculated similarity score greater than maximum, update the values
                if sim_score > max_sim_score:
                    max_sim_score = sim_score
                    best_left_indexes = left_indexes
                    best_right_indexes = right_indexes
                    threshold = val
                    feature = f

        # calculate the gain        
        gain = max_sim_score - parent_sim
        # if gain less than gamma value, prune the tree. It means stop the tree from growin any further for that sub-tree. 
        if gain <= self.gamma:
            return [None], [None], None, None
        
        return best_left_indexes, best_right_indexes, threshold, feature

    def predict(self,values):
        """
        values:pd.DataFrame -> feature values for prediction

        This function makes predictions based on given the given values.
        """
        result = []
        # loop through each observation
        for _,row in values.iterrows():
            # and start to travers the tree
            predicted_value = self.__traverse_tree(row,self.root)
            result.append(predicted_value)
        
        return np.array(result)

    def __traverse_tree(self,val,node):
        """
        val:pd.Series -> feature vector 
        node:Node -> current node
        
        This function recursively traverses the tree, and ouputs the prediction value.
        """
        if node.is_leaf:
            return np.sum(node.values).item() / (len(node.values) + self.regularization_lambda)

        # if value smaller than the value of the node, then go to the left
        if val[node.feature] <= node.threshold:
            return self.__traverse_tree(val,node.left)
        # if value greater than the value of the current node, then go to the right
        else:
            return self.__traverse_tree(val,node.right)
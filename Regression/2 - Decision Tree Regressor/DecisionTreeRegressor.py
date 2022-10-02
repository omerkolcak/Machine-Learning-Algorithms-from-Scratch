import numpy as np
import pandas as pd

class Node():

    def __init__(self,feature=None,threshold=None,left_child=None,right_child=None,values=None,variance=None,is_leaf=False):
        self.feature = feature
        self.threshold = threshold
        self.left = left_child
        self.right = right_child
        self.is_leaf = is_leaf
        self.values = values
        self.variance = variance

class DecisionTreeRegressor():

    def __init__(self,min_sample=1,max_depth=10,number_of_features=None):
        self.features = None
        self.number_of_features = number_of_features
        self.root = None
        self.min_sample = min_sample
        self.max_depth = max_depth


    def fit(self,x,y):
        self.features = x.columns
        self.root = self.__expand_tree(x,y,0)

    def predict(self,values):
        result = []
        for _,row in values.iterrows():
            predicted_value = self.__traverse_tree(row,self.root)
            result.append(predicted_value)
        
        return np.array(result)

    def __traverse_tree(self,val,node):
        if node.is_leaf:
            return np.mean(node.values)

        if val[node.feature] <= node.threshold:
            return self.__traverse_tree(val,node.left)
        else:
            return self.__traverse_tree(val,node.right)

    def __expand_tree(self,x,y,depth):
        if depth == self.max_depth or y.shape[0] <= self.min_sample:
            return  Node(values=y, variance=np.var(y), is_leaf=True) 

        features = self.features if self.number_of_features == None else np.random.choice(self.features,self.number_of_features,replace=False)

        left_idx,right_idx,threshold,feature = self.__find_the_attribute(x,y,features,np.var(y))

        left = self.__expand_tree(x.loc[left_idx,:],y.loc[left_idx,:],depth+1)
        right = self.__expand_tree(x.loc[right_idx,:],y.loc[right_idx,:],depth+1)

        return Node(feature,threshold,left,right,y,False)

    def __find_the_attribute(self,x,y,features,parent_variance):
        max_variance_reduction = -1000
        last_left_indexes = None
        last_right_indexes = None
        threshold = None
        feature = None

        n = y.shape[0]
        for f in features:
            unique_values = list(x[f].unique())

            for val in unique_values:
                left_indexes = x[x[f] <= val].index
                right_indexes = x[x[f] > val].index

                left_values = y.loc[left_indexes,:]
                right_values = y.loc[right_indexes,:]

                left_variance = np.var(left_values)*left_values.shape[0] / n  if len(left_indexes) > 0 else 0
                right_variance = np.var(right_values)*right_values.shape[0] / n if len(right_indexes) > 0 else 0

                variance_reduction = parent_variance - left_variance - right_variance 
                variance_reduction = variance_reduction.item()

                if variance_reduction > max_variance_reduction:
                    max_variance_reduction = variance_reduction
                    last_left_indexes = left_indexes
                    last_right_indexes = right_indexes
                    threshold = val
                    feature = f

        return last_left_indexes, last_right_indexes, threshold, feature

    def print_leaves(self,node):

        if node.is_leaf:
            print("values: ",node.values)
            print("variance: ",node.variance)
            return

        self.print_leaves(node.left)
        self.print_leaves(node.right)




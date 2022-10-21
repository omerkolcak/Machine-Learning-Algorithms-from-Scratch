import numpy as np
import pandas as pd

# calculates the purity of a set
def calculate_entropy(y):
    total_samples = y.shape[0]
    
    positive_samples = len(y[y[y.columns[0]] > 0]) / total_samples
    negative_samples = 1 - positive_samples
    
    return -positive_samples * np.log2(positive_samples) - negative_samples * np.log2(negative_samples)

# calculates the gini impurity
def calculate_gini_impurity(y):
    total_samples = y.shape[0]
    
    positive_samples = len(y[y[y.columns[0]] > 0]) / total_samples
    negative_samples = 1 - positive_samples
    
    return 1 - positive_samples**2 - negative_samples**2

class Node():

    def __init__(self,feature=None,threshold=None,left_child=None,right_child=None,values=None,is_leaf=False):
        self.feature = feature
        self.threshold = threshold
        self.left = left_child
        self.right = right_child
        self.is_leaf = is_leaf
        self.values = values
        
class DecisionTreeClassifier():

    def __init__(self,min_sample=1,max_depth=10,number_of_features=None,measure_of_disorder=None):
        self.root = None
        self.features = None
        self.number_of_features = number_of_features
        self.min_sample = min_sample
        self.max_depth = max_depth
        
        if measure_of_disorder == None:
            raise Exception("A measue of disorder must given")
            
        self.measure_of_disorder = measure_of_disorder # measure of disorder function

    def fit(self,x,y):
        self.features = x.columns
        self.root = self.expand_tree(x,y,0)

    def expand_tree(self,x,y,depth):
        # if reached to the max depth or after the splie there are less amount of sample or all the lables are the same return the leaf node
        if (depth == self.max_depth or y.shape[0] <= self.min_sample or len(y[y.columns[0]].unique()) == 1):
            return Node(values=y[y.columns[0]].value_counts().idxmax(), is_leaf=True) 

        # if number of features is not none randomly pick features 
        features = self.features if self.number_of_features == None else np.random.choice(self.features,self.number_of_features,replace=False)

        # find the best attribute and threshold by calculating the information gain
        left_idx,right_idx,threshold,feature = self.find_the_attribute(x,y,features)
        
        if len(left_idx) == 0 or len(right_idx) == 0:
            return Node(values=y[y.columns[0]].value_counts().idxmax(), is_leaf=True)

        left = self.expand_tree(x.loc[left_idx,:],y.loc[left_idx,:],depth+1) # recursively build left subtree
        right = self.expand_tree(x.loc[right_idx,:],y.loc[right_idx,:],depth+1) # recursively build right subtree

        return Node(feature,threshold,left,right,y,False) # return the root of the subtree, finally the main root of decision tree
    

    def find_the_attribute(self,x,y,features):
        max_information_gain = -10000
        last_left_indexes = None
        last_right_indexes = None
        threshold = None
        feature = None
        
        # get the measure of disorder before the split
        current_entropy = self.measure_of_disorder(y)

        n = y.shape[0]
        # loop through each features
        for f in features:
            unique_values = list(x[f].unique())

            # loop through each value for a feature
            for val in unique_values:
                # get the left and right indexes by the threshold
                left_indexes = x[x[f] <= val].index
                right_indexes = x[x[f] > val].index

                # get the left and right target variables
                left_values = y.loc[left_indexes,:]
                right_values = y.loc[right_indexes,:]
                
                # calculate the information gain based on the split
                ig = self.calculate_information_gain(current_entropy,left_values,right_values)
                
                # if calculated information gain is greater than the previous information gain, set the new one
                if ig > max_information_gain:
                    max_information_gain = ig
                    last_left_indexes = left_indexes
                    last_right_indexes = right_indexes
                    feature = f
                    threshold = val
      
        return last_left_indexes, last_right_indexes, threshold, feature
            
    def calculate_information_gain(self,cur_en,left,right):
        
        # it means there is no split, therefore no information gain
        if len(left) == 0 or len(right) == 0:
            return 0
        
        left_entropy = self.measure_of_disorder(left)
        right_entropy = self.measure_of_disorder(right) 
        
        total_samples = len(left) + len(right)
        
        ig = cur_en - left_entropy * len(left) / total_samples - right_entropy * len(right) / total_samples
        
        return ig

    def predict(self,values):
        result = []
        for _,row in values.iterrows():
            # traverse the tree by depth first search approach, and get the label of leaf value
            predicted_value = self.traverse_tree(row,self.root)
            result.append(predicted_value)
        
        return np.array(result)

    def traverse_tree(self,val,node):
        # if reached to the leaf, return the leaf label
        if node.is_leaf:
            return node.values

        # if smaller than the threshold of node go to the left of the tree
        if val[node.feature] <= node.threshold:
            return self.traverse_tree(val,node.left)
        # else go to the right of the tree
        else:
            return self.traverse_tree(val,node.right)


    def print_leaves(self,node):
        if node.is_leaf:
            print("label: ",node.values)
            return

        self.print_leaves(node.left)
        self.print_leaves(node.right)
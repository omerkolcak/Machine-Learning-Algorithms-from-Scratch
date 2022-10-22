import sys

sys.path.append("../3 - Decision Tree Classifier")

from DecisionTreeClassifier import *

class RandomForestClassifier():

    def __init__(self, number_of_estimators=10,min_sample=1,max_depth=100,number_of_features=None,measure_of_disorder=None):
        self.number_of_estimators = number_of_estimators
        self.number_of_features = number_of_features
        self.min_sample = min_sample
        self.max_depth = max_depth
        self.measure_of_disorder = measure_of_disorder
        self.trees = []

    def fit(self,x,y):
        # train all the decision trees independently
        for i in range(self.number_of_estimators):
            print(f"Tree {i+1} is being trained...")
            
            tree = DecisionTreeClassifier(min_sample=self.min_sample,max_depth=self.max_depth,
                                          number_of_features=self.number_of_features,measure_of_disorder=self.measure_of_disorder)
            
            # perform bootstrapping
            x_bootstraped, y_bootstraped = self.bootstraping(x,y) 

            # train the tree
            tree.fit(x_bootstraped,y_bootstraped)
            
            self.trees.append(tree)

    # apply bootstraping on dataset
    def bootstraping(self,x,y):
        number_of_samples = x.shape[0]

        # randomly select samples from the dataset, allow duplicates
        idx = np.random.choice(x.index,number_of_samples,replace=True)

        # reset index because duplicate rows have the same index
        x = x.loc[idx,:].reset_index(drop=True)
        y = y.loc[idx,:].reset_index(drop=True)

        return x,y

    def predict(self,values):
        all_predictions = []
        # perform prediction on the trained trees
        for tree in self.trees:
            all_predictions.append(tree.predict(values))

        # perform most voting system, get the most common label from the predictions
        preds = np.swapaxes(all_predictions,0,1)
        return np.argmax(np.apply_along_axis(np.bincount, 1, preds),1)
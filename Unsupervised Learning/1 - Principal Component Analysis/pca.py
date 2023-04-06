import numpy as np

class PCA():
    
    def __init__(self,k=None):
        self.principal_components = None
        self.k = k
    
    def fit_transform(self,X):
        # feature scaling
        X_normalized = (X - np.mean(X,axis=0)) / X.std(axis=0)
        # calculate covariance matrix
        m = X.shape[0]
        covariance_matrix = np.dot(X_normalized.T,X_normalized)
        # calculate singular value decomposition
        u,s,v = np.linalg.svd(covariance_matrix)
        
        # if k is not pre-defined, find the optimal value for k
        if self.k == None:
            self.k = self.find_optimal_k(s)
        
        # get the k principal components
        self.principal_components = u[:,:self.k]
        
        # return the transformation
        return np.dot(X_normalized,self.principal_components)
    
    def find_optimal_k(self,s):
        """
        Function to find optimal k.
        """
        s_sum = s.sum()
        for k in range(1,s.shape[0] + 1):
            if s[:k].sum() / s_sum >= 0.99:
                return k
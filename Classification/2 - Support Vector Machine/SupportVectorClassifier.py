import numpy as np
import random
from scipy.optimize import minimize, LinearConstraint, Bounds

def lagrange_problem(alpha,x,y):
    '''
    result = 0
    for i,ai in enumerate(alpha):
        for j,aj in enumerate(alpha):
            result += ai*aj*np.dot(y[i],y[j])*np.dot(x[i],x[j])

    result = np.sum(alpha) - 0.5 * result

    return result '''
    # vectorized version
    axy = np.dot(x.T,alpha*y)
    aaxxyy = np.dot(axy.T,axy)
    
    return np.sum(alpha) - 0.5*aaxxyy

class SupportVectorClassifier():
    def __init__(self, C=1, kernel_function=None, gamma=0.001,epsilon = 1e-7):
        self.C = C
        self.kernel = kernel_function
        self.gamma = gamma
        self.epsilon = epsilon
        self.train_data = None # it will be useful if kernel is used.
        self.w = None
        self.b = None
        self.alpha = None
        self.n = None
        self.m = None
        self.support_vectors = None
        self.support_labels = None
        
    def fit(self,x,y):
        
        self.m,self.n = x.shape
        
        y = np.where(y > 0,1,-1)
        y = y.reshape(-1,1)
        
        if self.kernel != None:
            self.trained_data = x
            x = np.apply_along_axis(lambda x1 : np.apply_along_axis(lambda x2:  self.kernel(x1, x2,self.gamma), 1, x),1, x)
        
        self.alpha = np.random.uniform(low=0.0, high=1.0, size=(self.m,))*self.C
        
        bounds_alpha = Bounds(np.zeros(self.m), np.full(self.m,self.C))
        linear_constraint = LinearConstraint(y.reshape(y.shape[0]),[0],[0]) # alpha.dot(y) = 0
        
        res = minimize(
            fun=lambda a : -lagrange_problem(a,x,y.reshape(y.shape[0])), #what we want to minimize
            x0=self.alpha,
            constraints=[linear_constraint],bounds=bounds_alpha,
            method='SLSQP',options={'disp': True,'maxiter' : 10000})
        
        self.alpha = res.x
        
        self.w = np.dot(self.alpha.T,x*y)
        
        self.support_vectors = x[self.alpha > self.epsilon]
        self.support_labels = y[self.alpha > self.epsilon]
        b = self.support_labels[0] - np.matmul(self.support_vectors[0].T, self.w)
        self.b = b[0]
        
    def predict(self,x):
        if self.kernel != None:
            x = np.apply_along_axis(lambda x1 : np.apply_along_axis(lambda x2:  self.kernel(x1, x2,self.gamma), 1, self.trained_data),1, x)
        
        return np.where(np.matmul(x, self.w) + self.b > 0,1,0)
        
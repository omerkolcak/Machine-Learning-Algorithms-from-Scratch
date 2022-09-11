import numpy as np
import random

def mean_squared_error(pred,y):
    return np.sum((pred - y)**2) / y.shape[0]

class LinearRegressor():

    def __init__(self, learning_rate=0.01, iterations=100):
        self.w = None
        self.b = None
        self.m = None
        self.learning_rate = learning_rate
        self.iterations = iterations
    
    def fit(self,x,y):
        self.m = x.shape[0]
        self.n = x.shape[1]
        self.w = np.random.uniform(low=0, high=1, size=(self.n,))
        self.b = random.uniform(0, 1)

        for iter in range(self.iterations):
            yhat = np.dot(x,self.w) + self.b # (m,n) * (n,1) -> (m,1)

            d_w = np.dot(x.T,yhat - y) / self.m # (n,m) * (m,1) -> (n,1)
            d_b = np.sum(yhat - y) / self.m

            self.w -= self.learning_rate * d_w
            self.b -= self.learning_rate * d_b

            if iter % 50 == 0:
                mse = mean_squared_error(self.predict(x),y)
                print(f"{iter}. Iteration Training loss: {mse}")


    
    def predict(self,x):
        return np.dot(x,self.w) + self.b
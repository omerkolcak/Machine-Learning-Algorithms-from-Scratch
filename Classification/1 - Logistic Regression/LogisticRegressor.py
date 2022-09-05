import numpy as np
import random

def log_loss(probs,y):
    return np.sum(-y*np.log(probs + 0.00001) - (1-y)*np.log(1+0.00001-probs)) / y.shape[0]

def sigmoid_function(theta):
    probs = 1 / (1 + np.exp(-theta))
    return probs

class LogisticRegressor():

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
            yhat = self.predict(x)

            d_w = np.dot(x.T,yhat - y) / self.m # (n,m) * (m,1) -> (n,1)
            d_b = np.sum(yhat - y) / self.m

            self.w -= self.learning_rate * d_w
            self.b -= self.learning_rate * d_b

            if iter % 50 == 0:
                theta = np.dot(x,self.w) + self.b # (m,n) * (n,1) -> (m,1)
                probs = sigmoid_function(theta)

                loss = log_loss(probs,y)
                print(f"{iter}. Iteration Training loss: {loss}")


    def predict(self,x):
        theta = np.dot(x,self.w) + self.b # (m,n) * (n,1) -> (m,1)
        probs = 1 / (1 + np.exp(-theta))

        return [1 if p >= 0.5 else 0 for p in probs]

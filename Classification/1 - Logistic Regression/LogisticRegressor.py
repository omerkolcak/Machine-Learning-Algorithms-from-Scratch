import numpy as np
import random

def sigmoid_function(theta):
    probs = 1 / (1 + np.exp(-theta))
    return probs

class LogisticRegressor():

    def __init__(self, learning_rate=0.01, iterations=100,reg_lambda=0):
        self.w = None
        self.b = None
        self.m = None
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.reg_lambda = reg_lambda
    
    def fit(self,x,y):
        self.m = x.shape[0]
        self.n = x.shape[1]
        self.w = np.random.uniform(low=0, high=1, size=(self.n,))
        self.b = random.uniform(0, 1)

        for iter in range(self.iterations):

            if iter % 50 == 0:
                theta = np.dot(x,self.w) + self.b # (m,n) * (n,1) -> (m,1)
                probs = sigmoid_function(theta)

                loss = self.log_loss(probs,y,self.reg_lambda)
                print(f"{iter}. Iteration Training loss: {loss}")
                print(self.w,self.b)

            yhat = self.predict(x)

            d_w = np.dot(x.T,yhat - y) / self.m - ((self.reg_lambda * self.w) / self.m) # (n,m) * (m,1) -> (n,1)
            d_b = np.sum(yhat - y) / self.m

            self.w -= self.learning_rate * d_w
            self.b -= self.learning_rate * d_b



    def predict(self,x):
        theta = np.dot(x,self.w) + self.b # (m,n) * (n,1) -> (m,1)
        probs = 1 / (1 + np.exp(-theta))

        return [1 if p >= 0.5 else 0 for p in probs]

    def log_loss(self,probs,y,reg_lambda=0):
        return np.sum(-y*np.log(probs + 0.00001) - (1-y)*np.log(1+0.00001-probs)) / y.shape[0] + (reg_lambda * np.sum(np.square(self.w))) / (2 * self.m)

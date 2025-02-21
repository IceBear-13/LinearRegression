import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, data, L, epochs):
        self.data = pd.read_csv(data)
        self.m = 0
        self.b = 0
        self.L = L
        self.epochs = epochs
        
    def loss_function(self):
        X = self.data['X']
        y = self.data['y']
        return np.mean((y - (self.m * X + self.b)) ** 2) 
        
    def gradient_descent(self):
        N = len(self.data)
        X = self.data['X']
        y = self.data['y']
        
        m_gradient = -(2/N) * sum(X * (y - (self.m * X + self.b)))
        b_gradient = -(2/N) * sum(y - (self.m * X + self.b))
        
        self.m = self.m - self.L * m_gradient
        self.b = self.b - self.L * b_gradient
        
    def train(self):
        for i in range(self.epochs):
            self.gradient_descent()
            if i % 100 == 0:
                print(f'Epoch {i}: Loss = {self.loss_function()}')
                
    def plot_regression_line(self):
        X = self.data['X']
        y = self.data['y']
        plt.scatter(X, y)
        plt.plot(X, self.m * X + self.b, color='red')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.show()
        
    def predict(self, X):
        return self.m*X + self.b

# Example usage
model = LinearRegression('C:/Users/User/Desktop/projects/pytorch/data/data.csv', 0.00000001, 1000)
model.train()
model.plot_regression_line()
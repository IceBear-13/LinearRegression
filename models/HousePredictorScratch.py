import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class HousePredictor:
    def __init__(self, dataSrc, L, epochs):
        self.data = pd.read_csv(dataSrc)
        self.L = L
        self.epochs = epochs
        self.m = 0
        self.b = 0
        
    def loss_function(self):
        bedrooms = self.data['Bedrooms']
        bathrooms = self.data['Bathrooms']
        area = self.data['SquareFeet']
        dis = self.data['ProximityToCityCenter']
        age = self.data['Age']
        price = self.data['Price']
        
        return np.mean((price - (self.m*(bedrooms + bathrooms + area - dis - age) + self.b) ) ** 2)
    
    def gradient_descent(self):
        N = len(self.data)
        bedrooms = self.data['Bedrooms']
        bathrooms = self.data['Bathrooms']
        area = self.data['SquareFeet']
        dis = self.data['ProximityToCityCenter']
        age = self.data['Age']
        price = self.data['Price']
        
        m_gradient = -(2/N) * sum((bedrooms + bathrooms + area - dis - age) * (price - (self.m * (bedrooms + bathrooms + area - dis - age)) + self.b))
        b_gradient = -(2/N) * sum(price - (self.m * (bedrooms + bathrooms + area - dis - age) + self.b))
        
        self.m = self.m - self.L * m_gradient
        self.b = self.b - self.L * b_gradient
        
        
    def train(self):
        for i in range(self.epochs):
            self.gradient_descent()
            if i % 100 == 0:
                print(f'Epoch {i}: Loss = {self.loss_function()}')

    def plot(self):
        plt.scatter(self.data['SquareFeet'], self.data['Price'], color='blue', label='Actual Price')
        predicted_price = self.m * (self.data['Bedrooms'] + self.data['Bathrooms'] + self.data['SquareFeet'] - self.data['ProximityToCityCenter'] - self.data['Age']) + self.b
        plt.plot(self.data['SquareFeet'], predicted_price, color='red', label='Predicted Price')
        plt.xlabel('Square Feet')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        
    def predict_price(self, bedroom, bathroom, area, dist, age):
        scaled_features = self.scaler.transform([[bedroom, bathroom, area, dist, age]])
        return self.m*(scaled_features[0][0] + scaled_features[0][1] + scaled_features[0][2] - scaled_features[0][3] - scaled_features[0][4]) + self.b
    

# Sample usage
model = HousePredictor('C:/Users/User/Desktop/projects/LLMs/data/house_prices.csv', 0.00000015, 10000)
model.train()
model.plot()
print(model.predict_price(4,1,1413,13,50))
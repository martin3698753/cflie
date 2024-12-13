import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(x, 0)

def relu_prime(x):
    return x > 0

def window(power, battery, n=10):
    x, y = [], []
    for i in range(len(power) - n):
        # Create input sequence of power measurements
        x.append(power[i:i+n])
        # Target is the next battery voltage
        y.append(battery[i+n])
    x = np.array(x)
    y = np.array(y)
    x = x.reshape(x.shape[0], -1)
    return x, y

class Lin(nn.Module):
    def __init__(self):
        super(Lin, self).__init__()
        self.lin1 = nn.Linear(10, 20)
        self.tanh1 = nn.Tanh()
        self.lin2 = nn.Linear(20, 1)
        self.tanh2 = nn.Tanh()
    def forward(self, x):
        out = self.lin1(x)
        out = self.tanh1(out)
        out = self.lin2(out)
        out = self.tanh2(out)
        return out

def train_lin(x, y):
    X_train = torch.from_numpy(x).float()
    y_train = torch.from_numpy(y).float()
    model = Lin()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    num_epochs = 5
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        print(y_train.shape, y_train.dtype)
        print(outputs)
        loss = criterion(outputs, y_train)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def predict_lin(s):
    s = torch.from_numpy(s).float()
    model = Lin()
    o = np.empty(0)
    with torch.no_grad():
        model.eval() #set model to eval mode
        for data in s:
            output = model(data) #single number
            o = np.append(o, output)

    return o



class LNU:
    def __init__(self, input_size=10):
        self.w1 = np.random.rand(5, input_size)
        self.b1 = np.random.rand(5, 1) - 0.5
        self.w2 = np.random.rand(1, 5) - 0.5
        self.b2 = np.random.rand(1, 1) - 0.5


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import maketab as mt

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input to hidden layer
        self.fc2 = nn.Linear(hidden_size, output_size)  # Hidden to output layer
        self.relu = nn.ReLU()  # Activation function
        self.prelu = nn.PReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.fc1(x)  # Pass through the hidden layer
        #out = self.relu(out)  # Apply ReLU activation
        out = self.tanh(out)
        out = self.fc2(out)  # Pass through the output layer
        return out

# Load the checkpoint
checkpoint = torch.load('bat_model.pth')

# Extract architecture parameters
seq_length = checkpoint['seq_length']
num_layers = checkpoint['num_layers']
starting_point = checkpoint['starting_point']
input_size = checkpoint['input_size']
hidden_size = checkpoint['hidden_size']
output_size = checkpoint['output_size']

# Instantiate the model
model = MLP(input_size, hidden_size, output_size)

# Load the state dictionary
model.load_state_dict(checkpoint['state_dict'])

def eval(x, t):
    model.eval()
    with torch.no_grad():
        slope2, slope, intercept = np.polyfit(t, x, 2)
        mean = np.mean(x)
        std = np.std(x)
        X = torch.tensor((slope2, slope, intercept, mean, std), dtype=torch.float32)
        pred = model(X).numpy()
        reg = pred[0]*t**2 + pred[1]*t + pred[2]
        print(f"result: {reg[0]}")

def pred(x):
    t = np.linspace(0, 1, seq_length)
    eval(x, t)

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class BatSeqModel:
    def __init__(self, checkpoint_path='bat_seq_model.pth'):
        self.checkpoint = torch.load(checkpoint_path)
        self.input_size = self.checkpoint['input_size']
        self.hidden_size = self.checkpoint['hidden_size']
        self.output_size = self.checkpoint['output_size']
        self.n = self.checkpoint['seq_length']
        self.norm_upper = self.checkpoint['norm_upper']
        self.norm_lower = self.checkpoint['norm_lower']
        self.model = MLP(self.input_size, self.hidden_size, self.output_size)
        self.model.load_state_dict(self.checkpoint['state_dict'])
        self.memory = np.empty(0)
        self.g = np.empty(0)  # Store predictions
        self.signal = np.empty(0)  # Store input signals

    def norm(self, signal):
        signal = (signal - self.norm_lower) / (self.norm_upper - self.norm_lower)
        return signal

    def eval(self, data):
        data = self.norm(data)
        test_input = torch.tensor(data, dtype=torch.float32)
        with torch.no_grad():
            pred = self.model(test_input).squeeze().numpy()
            self.g = np.append(self.g, pred)  # Append prediction to g
            self.signal = np.append(self.signal, data)  # Append input data to signal
            #
            #
            print(f"prediction: {pred.mean()}")
            self.write_to_file(pred)

    def pred(self, x):
        self.memory = np.append(self.memory, x)
        if len(self.memory) >= self.n:
            self.eval(self.memory)
            self.memory = np.empty(0)

    def done(self):
        t = np.arange(len(self.signal))  # Create time axis
        plt.plot(t, self.signal, label='Signal')  # Plot signal
        plt.plot(t, self.g, label='Predictions')  # Plot predictions
        plt.legend()
        plt.show()

    def write_to_file(self, pred):
        filename = 'pred.csv'
        if not os.path.exists(filename):
            f = open(filename, 'w')
            f.write('predictions'+'\n')
            f.close()

        f = open(filename, 'a')
        for i in pred:
            f.write(str(i)+'\n')
        f.close()

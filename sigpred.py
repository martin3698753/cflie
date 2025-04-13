import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import maketab as mt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
plt.rcParams['mathtext.fontset'] = 'cm'
import gc


sec_norm = 410
cutoff = 50 #cutting of time it takes to lift off
train_dir = (['data/31-1-25/'])
test_dir = (['data/21-2-25/'])
sig_norm = [3.7, 2.3]

def norm(signal, norm):
    signal = (signal - norm[1]) / (norm[0] - norm[1])
    return signal

def load_data(path_dir):
    t, signal = mt.battery(path_dir)
    signal = norm(signal[cutoff:], sig_norm)
    return signal

def make_data(train_dir):
    if(len(train_dir) == 1):
        for d in train_dir:
            train_data = load_data(d)
        #plt.plot(train_data[0])
        #plt.plot(train_data[1])
        #plt.show()
        return train_data
    else:
        #train_data = np.empty((len(train_dir), 0))
        train_data = load_data(train_dir[0])
        for d in train_dir[1:]:
            single_data = load_data(d)
            train_data = np.concatenate((train_data, single_data), axis=1)
        #plt.plot(train_data[0])
        #plt.plot(train_data[1])
        #plt.plot(train_data[2])
        #plt.show()
        return train_data

class SequenceDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.sequence_length = sequence_length

    def __len__(self):
        # The number of possible sequences is len(data) - 2 * sequence_length + 1
        return len(self.data) - 2 * self.sequence_length + 1

    def __getitem__(self, idx):
        input_seq = self.data[idx : idx + self.sequence_length]
        output_seq = self.data[idx + self.sequence_length : idx + 2 * self.sequence_length]
        return input_seq, output_seq

class FeedforwardModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedforwardModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        return x

def train(num_epochs, dataloader, model, criterion, optimizer):
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0

        for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(input_seq)
            loss = criterion(output, target_seq)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

            # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')

def predict(data, seq_length, theta, model):
    model.eval()
    with torch.no_grad():
        pred = np.empty(0)
        for i in range(0, len(data) - seq_length + 1, seq_length*(theta+1)):
            X = torch.tensor(data[i:i+seq_length], dtype=torch.float32).unsqueeze(0)
            y = model(X)
            y_pred = y.squeeze(0).numpy()
            pred = np.append(pred, y_pred)
            for j in range(theta):
                y = model(y)
                y_pred = y.squeeze(0).numpy()
                pred = np.append(pred, y_pred)
                # if(len(data) <= i+(2*30*(j+1))):
                #     break
            plt.axvline(i+seq_length, linestyle=":", color="black", linewidth=0.5)

        pred = np.append(np.zeros(seq_length), pred)

        new_pred = pred[seq_length:len(data)]
        new_data = data[seq_length:]
        plt.plot(new_data)
        plt.plot(new_pred)
        plt.show()
        mse = mean_squared_error(new_data, new_pred)
        r2 = r2_score(new_data, new_pred)
        return mse, r2

def test_model(seq_length, hidden_size, num_epochs, batch_size, learning_rate, theta):
    train_data = make_data(train_dir)
    dataset = SequenceDataset(train_data, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = FeedforwardModel(seq_length, hidden_size, seq_length)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train(num_epochs, dataloader, model, criterion, optimizer)
    mse, r2 = predict(train_data, seq_length, theta, model)
    print(mse, r2)


if __name__ == "__main__":
    seq_length = 50
    hidden_size = 80
    learning_rate = 0.0001
    num_epochs = 200
    batch_size = 32
    theta = 150

    test_model(seq_length, hidden_size, num_epochs, batch_size, learning_rate, theta)



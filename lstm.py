import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def scale_df(df):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_df = scaler.fit_transform(df)
    return scaled_df

def prepare_data(df, n_steps):
    df = dc(df)

    df.set_index('time', inplace=True)
    for i in range(1, n_steps+1):
        df[f'y(t-{i})'] = df['y'].shift(i)

    df.dropna(inplace=True)
    return df

def train_one_epoch(epoch, model, train_loader, optimizer, loss_function):
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 99:  # print every 100 batches
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
                                                    avg_loss_across_batches))
            running_loss = 0.0
    print()

def validate_one_epoch(model, test_loader, loss_function):
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)

    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('***************************************************')
    print()


def show(X, y, model):
    with torch.no_grad():
        predicted = model(X.to(device)).to('cpu').numpy()
        
    plt.plot(y, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def setup(t, y):
    lookback = 10 #Window size
    
    df = pd.DataFrame({
        'time':t,
        'y':y,
    })
    
    df = prepare_data(df, lookback)
    df = scale_df(df)
    
    X = df[:, 1:]
    y = df[:, 0]
    
    X = dc(np.flip(X, axis=1))
    split_index = int(len(X) * 0.75)
    
    X_train = X[:split_index]
    X_test = X[split_index:]
    
    y_train = y[:split_index]
    y_test = y[split_index:]
    
    X_train = X_train.reshape((-1, lookback, 1))
    X_test = X_test.reshape((-1, lookback, 1))
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    
    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()
    
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    batch_size = 16
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return X_train, y_train, X_test, y_test, train_loader, test_loader

def init(t, y):
    model = LSTM(1, 4, 1)
    model.to(device)

    X_train, y_train, X_test, y_test, train_loader, test_loader = setup(t, y)
    
    learning_rate = 0.001
    num_epochs = 50
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
    for epoch in range(num_epochs):
        train_one_epoch(epoch, model, train_loader, optimizer, loss_function)
        validate_one_epoch(model, test_loader, loss_function)
    
    show(X_test, y_test, model)

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
PATH = 'bat_pics/motor/lnu/'
import time


sec_norm = 410
cutoff = 50 #cutting of time it takes to lift off
train_dir = (['data/24-1-25/', 'data/4-2-25/', 'data/5-2-25/'])
test_dir = (['data/31-1-25/', 'data/11-4-25/', 'data/9-4-25/'])
sig_norm = [3.7, 2.3]

def norm(signal, norm):
    signal = (signal - norm[1]) / (norm[0] - norm[1])
    return signal

def load_data(path_dir):
    t, signal = mt.battery(path_dir)
    me = mt.power(path_dir)
    secleft = t[-1]/1000
    tleft = 1 - t / max(t)
    tleft = tleft*(secleft/sec_norm)
    signal = norm(signal[cutoff:], sig_norm)
    tleft = tleft[cutoff:]
    t = t[cutoff:]
    me = me[cutoff:]
    normalized_train = np.array([signal, me, tleft])
    return normalized_train

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

class FunctionDataset(Dataset):
    def __init__(self, data, n):
        self.g = data[0]  # Function g
        self.h = data[1]
        self.f = data[2]  # Function f
        self.n = n  # Sequence length
        self.total_length = len(self.g)

    def __len__(self):
        # The number of sequences is the total length minus the sequence length plus 1
        return self.total_length - self.n + 1

    def __getitem__(self, idx):
        # Get a sequence of g and the corresponding sequence of f
        g_seq = self.g[idx:idx + self.n]
        f_seq = self.f[idx:idx + self.n]
        h_seq = self.h[idx:idx + self.n]

        # Convert to PyTorch tensors
        g_tensor = torch.tensor(g_seq, dtype=torch.float32)
        h_tensor = torch.tensor(h_seq, dtype=torch.float32)
        f_tensor = torch.tensor(f_seq, dtype=torch.float32)
        in_tensor = torch.cat((g_tensor, h_tensor))

        return in_tensor, f_tensor

def create_dataloader(data, seq_length, batch_size=1, shuffle=False):
    dataset = FunctionDataset(data, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.register_buffer("hardcoded_matrix", torch.ones(output_size, 1))

    def forward(self, x):
        out = self.fc1(x)
        out = out @ self.hardcoded_matrix.t()
        return out

def train(num_epochs, dataloader, model, criterion, optimizer):
    epoch_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0  # To accumulate loss for the entire epoch
        num_batches = 0
        for batch_X, batch_Y in dataloader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        avg_loss = (epoch_loss / num_batches)
        epoch_losses.append(avg_loss)

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.8f}')
    # plt.plot(epoch_losses)
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.show()

def evaluate_and_plot(data, model, n, name, graph=True):
    model.eval()  # Set the model to evaluation mode

    g = data[0]
    h = data[1]
    f = data[2]

    predictions = []

    with torch.no_grad():  # Disable gradient computation
        for i in range(0, len(g)-n, n):
            g_seq = g[i:i+n]
            h_seq = h[i:i+n]
            g_tensor = torch.tensor(g_seq, dtype=torch.float32)
            h_tensor = torch.tensor(h_seq, dtype=torch.float32)
            X = torch.cat((g_tensor, h_tensor)).unsqueeze(0)
            # Convert to PyTorch tensor and add batch dimension
            #X = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)
            y = model(X)
            predictions.extend(y.squeeze(0).numpy())  # Remove batch dimension and convert to numpy

    predictions = np.array(predictions)
    # Plot the results
    plt.figure(figsize=(10, 6))
    t = np.arange(0, len(g))
    t = t/10
    plt.plot(t, g, label=r'$u(t)$')
    t = np.arange(0, len(h))
    t = t/10
    plt.plot(h, label=r'$m(t)$')
    t = np.arange(0, len(f))
    t = t/10
    plt.plot(t, f, label=r'$\hat{\tau}(t)$')
    t = np.arange(0, len(predictions))
    t = t/10
    plt.plot(t, predictions, label=r'$y(t)$')

    mse = mean_squared_error(f[:len(predictions)], predictions)
    r2 = r2_score(f[:len(predictions)], predictions)
    # text = (
    #     f"{'MSE':<5} {mse:>.4e}\n"
    #     f"{'R2':<5} {r2:>.4e}"
    # )

    # plt.text(
    #     0.0, 0.0,
    #     text,
    #     fontsize=11,
    #     fontfamily="monospace",  # Use a monospaced font
    #     verticalalignment="center",  # Align text vertically
    #     horizontalalignment="left",  # Align text horizontally
    #     bbox=dict(facecolor="lightgray", alpha=0.8, edgecolor="black"),  # Add a background box
    # )

    # Add labels and legend
    if graph is True:
        plt.xlabel('t(s)')
        plt.legend(fontsize=16)
        plt.grid(True)
        #plt.show()
        plt.savefig(PATH+name+'.pdf')
        plt.close()
    return mse, r2


def test_model(n, k, num_epochs, batch_size, learning_rate):
    # # Number of values to predict at a time
    # n = 30
    # # Hidden size
    # k = 50
    # num_epochs = 10
    # batch_size = 16
    # learning_rate = 0.001

    model = MLP(2*n, k, n)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_data = make_data(train_dir)
    test_data = make_data(test_dir)
    dataloader = create_dataloader(train_data, n, batch_size=batch_size, shuffle=True)
    start_time = time.time()
    train(num_epochs, dataloader, model, criterion, optimizer)
    end_time = time.time()
    print(f"Training completed in {(end_time - start_time):.4f} seconds")
    mse1, r21 = evaluate_and_plot(train_data, model, n, 'train'+str(n))
    mse2, r22 = evaluate_and_plot(test_data, model, n, 'test'+str(n), graph=False)
    for i, d in enumerate(test_dir):
        data = load_data(d)
        evaluate_and_plot(data, model, n, 'test'+str(i)+str(n)+str(k))

    # Force cleanup (optional)
    del model, criterion, optimizer
    gc.collect()  # Garbage collect to ensure no lingering references

    return n, learning_rate, mse1, mse2, r21, r22, (end_time - start_time)

if __name__ == "__main__":
    rows = []
    columns = ['n', 'lr', 'train_mse', 'test_mse', 'train_r2', 'test_r2', 'time']
    # rows.append(test_model(10, 1, 100, 16, 0.001))
    # rows.append(test_model(20, 1, 100, 16, 0.001))
    rows.append(test_model(30, 1, 50, 16, 0.0001))
    # rows.append(test_model(40, 1, 100, 16, 0.001))
    # rows.append(test_model(60, 1, 100, 16, 0.001))
    # rows.append(test_model(80, 1, 100, 16, 0.001))

    df = pd.DataFrame(rows, columns=columns)
    scale_cols = ['train_mse', 'test_mse', 'train_r2', 'test_r2']
    df[scale_cols] = (df[scale_cols] * 1000).round(2)
    df['time'] = df['time'].round(1)

    print(df)
    df['lr'] = '10^{-3}'
    print("    ")
    print("\n".join("&".join(f"{{${val}$}}" for val in row) + "\\\\" for _, row in df.iterrows()))

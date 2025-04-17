import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import maketab as mt
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import mean_squared_error, r2_score
plt.rcParams['mathtext.fontset'] = 'cm'
import gc
PATH = 'bat_pics/future/mlp/'
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
    signal = norm(signal[cutoff:], sig_norm)
    return signal

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

    def forward(self, x):
        x = self.fc1(x)
        x = self.sig(x)
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
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.8f}')

def predict(data, seq_length, theta, model, name, graph=True):
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
            plt.axvline((i+seq_length)/10, linestyle=":", color="black", linewidth=0.7)

        pred = np.append(np.zeros(seq_length), pred)

        new_pred = pred[seq_length:len(data)]
        new_data = data[seq_length:]
        t = np.arange(0, len(new_data))
        t = t/10
        if graph is True:
            plt.plot(t, new_data, label=r'$\hat{u}(t)$')
            plt.plot(t, new_pred, label=r'$y(t)$')
            plt.axvline((len(data))/10, linestyle=":", color="black", linewidth=0.7, label=fr"$\upsilon n; \upsilon = {theta}$")
            plt.xlabel('t(s)')
            plt.legend(fontsize=16)
            plt.grid(True)
            plt.savefig(PATH+name+'.pdf')
            plt.close()
        # mse = mean_squared_error(new_data, new_pred)
        # r2 = r2_score(new_data, new_pred)
        return new_data, new_pred


def test_model(seq_length, hidden_size, num_epochs, batch_size, learning_rate, theta):
    train_data1 = load_data(train_dir[0])
    dataset1 = SequenceDataset(train_data1, seq_length)
    train_data2 = load_data(train_dir[1])
    dataset2 = SequenceDataset(train_data2, seq_length)
    train_data3 = load_data(train_dir[2])
    dataset3 = SequenceDataset(train_data3, seq_length)

    combined_dataset = ConcatDataset([dataset1, dataset2, dataset3])
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    test_data1 = load_data(test_dir[0])
    test_data2 = load_data(test_dir[1])
    test_data3 = load_data(test_dir[2])

    model = FeedforwardModel(seq_length, hidden_size, seq_length)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start_time = time.time()
    train(num_epochs, dataloader, model, criterion, optimizer)
    end_time = time.time()
    print(f"Training completed in {(end_time - start_time):.4f} seconds")

    mse1 = np.zeros(2)
    r21 = np.zeros(2)
    mse2 = np.zeros(2)
    r22 = np.zeros(2)
    for i, th in enumerate(theta):
        data1, pred1 = predict(train_data1, seq_length, th, model, 'train1_'+str(seq_length)+str(hidden_size)+str(th), graph=True)
        data2, pred2 = predict(train_data2, seq_length, th, model, 'train2_'+str(seq_length)+str(hidden_size)+str(th), graph=True)
        data3, pred3 = predict(train_data3, seq_length, th, model, 'train3_'+str(seq_length)+str(hidden_size)+str(th), graph=True)

        data = np.concatenate([data1, data2, data3])
        pred = np.concatenate([pred1, pred2, pred3])

        mse1[i] = mean_squared_error(data, pred)
        r21[i] = r2_score(data, pred)


        data1, pred1 = predict(test_data1, seq_length, th, model, 'test1_'+str(seq_length)+str(hidden_size)+str(th), graph=True)
        data2, pred2 = predict(test_data2, seq_length, th, model, 'test2_'+str(seq_length)+str(hidden_size)+str(th), graph=True)
        data3, pred3 = predict(test_data3, seq_length, th, model, 'test3_'+str(seq_length)+str(hidden_size)+str(th), graph=True)

        data = np.concatenate([data1, data2, data3])
        pred = np.concatenate([pred1, pred2, pred3])

        mse2[i] = mean_squared_error(data, pred)
        r22[i] = r2_score(data, pred)

    del model, criterion, optimizer
    gc.collect()

    return seq_length, hidden_size, learning_rate, mse1[0], mse2[0], r21[0], r22[0], mse1[1], mse2[1], r21[1], r22[1], (end_time - start_time)


if __name__ == "__main__":
    rows = []
    columns = ['n', 'k', 'lr', 'train_mse_t1', 'test_mse_t1', 'train_r2_t1', 'test_r2_t1', 'train_mse_t2', 'test_mse_t2', 'train_r2_t2', 'test_r2_t2', 'time']
    #test_model(seq_length, hidden_size, num_epochs, batch_size, learning_rate, theta)
    rows.append(test_model(50, 1, 20, 16, 0.0001, [5, 10]))

    df = pd.DataFrame(rows, columns=columns)
    scale_cols = ['train_mse_t1', 'test_mse_t1', 'train_r2_t1', 'test_r2_t1', 'train_mse_t2', 'test_mse_t2', 'train_r2_t2', 'test_r2_t2']
    df[scale_cols] = (df[scale_cols] * 1000).round(2)
    df['time'] = df['time'].round(1)

    df.insert(3, 'sigma', r'\text{ReLU}')

    print(df)
    df['lr'] = '10^{-3}'
    print("    ")
    print("\n".join("&".join(f"{{${val}$}}" for val in row) + "\\\\" for _, row in df.iterrows()))

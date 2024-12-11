import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
    num_epochs = 100
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
    o = np.empty(0)
    with torch.no_grad():
        model.eval() #set model to eval mode
        for data in s:
            output = model(data) #single number
            o = np.append(o, output)

    return o



class LNU:
    def __init__(self, n=10):
        self.n = n
        self.network = nn.Sequential(
            nn.Linear(self.n, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )

    def prepare(self, power, battery):
        x, y = window(power, battery)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        return train_loader, X_test_tensor, y_test

    def train(self, power, battery, epochs=50):
        train_loader, X_test_tensor, y_test = self.prepare(power, battery)

        model = self.network
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        model.train()
        train_losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(batch_X)

                # Compute loss
                loss = criterion(outputs, batch_y)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Store average epoch loss
            train_losses.append(epoch_loss / len(train_loader))
            print(epoch, " : ", epoch_loss)

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test_tensor).numpy()
            mse = np.mean((test_predictions - y_test)**2)
            mae = np.mean(np.abs(test_predictions - y_test))

        return {
            'model': model,
            'predictions': test_predictions,
            'actual': y_test,
            'mse': mse,
            'mae': mae,
            'train_losses': train_losses
        }
    def visual(self, result):
        # Visualize results
        plt.figure(figsize=(12, 5))

        # Training Loss
        plt.subplot(1, 2, 1)
        plt.plot(result['train_losses'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        # Predictions vs Actual
        plt.subplot(1, 2, 2)
        plt.plot(result['actual'], label='Actual Voltage')
        plt.plot(result['predictions'], label='Predicted Voltage', linestyle='--')
        plt.title('Battery Voltage: Actual vs Predicted')
        plt.xlabel('Sample Index')
        plt.ylabel('Battery Voltage (V)')
        plt.legend()

        plt.tight_layout()
        plt.show()

        return result


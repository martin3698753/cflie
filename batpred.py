import torch
import torch.nn as nn
import numpy as np
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
    def __init__(self, checkpoint_path='final_model.pth'):
        self.checkpoint = torch.load(checkpoint_path)
        self.input_size = self.checkpoint['input_size']
        self.hidden_size = self.checkpoint['hidden_size']
        self.output_size = self.checkpoint['output_size']
        self.n = self.checkpoint['seq_length']
        self.norm_upper = self.checkpoint['sig_norm'][0]
        self.norm_lower = self.checkpoint['sig_norm'][1]
        self.sec_norm = self.checkpoint['sec_norm']
        self.model = MLP(self.input_size, self.hidden_size, self.output_size)
        self.model.load_state_dict(self.checkpoint['state_dict'])
        self.model.eval()

        self.u_memory = np.empty(0)
        self.m_memory = np.empty(0)
        self.timestamps = []
        self.g = np.empty(0)
        self.signal = np.empty(0)

    def normalize_u(self, signal):
        return (signal - self.norm_lower) / (self.norm_upper - self.norm_lower)

    def run_model(self, u_seq, m_seq, timestamp):
        u_seq = self.normalize_u(np.array(u_seq))
        m_seq = np.array(m_seq)
        data = np.concatenate((u_seq, m_seq))
        test_input = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = self.model(test_input).squeeze().numpy()
            pred_sec = pred * self.sec_norm
            self.g = np.append(self.g, pred_sec)
            self.signal = np.append(self.signal, u_seq)
            print(f"Zbývající čas letu: {pred_sec.mean():.2f} s")
            self.write_to_file(timestamp, pred_sec)

    def pred(self, u, m, timestamp):
        self.u_memory = np.append(self.u_memory, u)
        self.m_memory = np.append(self.m_memory, m)
        self.timestamps.append(timestamp)
        if len(self.u_memory) >= self.n and len(self.m_memory) >= self.n:
            self.run_model(self.u_memory[:self.n], self.m_memory[:self.n], self.timestamps[0])
            self.u_memory = self.u_memory[self.n:]
            self.m_memory = self.m_memory[self.n:]
            self.timestamps = self.timestamps[self.n:]

    def done(self):
        import matplotlib.pyplot as plt
        t = np.arange(len(self.signal))
        plt.plot(t, self.signal, label='Signal')
        plt.plot(t, self.g, label='Predictions (s)')
        plt.legend()
        plt.show()

    def write_to_file(self, timestamp, pred):
        filename = 'pred.csv'
        mode = 'a' if os.path.exists(filename) else 'w'
        with open(filename, mode) as f:
            if mode == 'w':
                f.write('timestamp,predictions_seconds\n')
            for val in pred:
                f.write(f"{timestamp},{val}\n")

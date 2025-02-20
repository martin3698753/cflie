import numpy as np
import maketab as mt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import maketab
import pickdir
import torch

def norm(s):
    scaler = MinMaxScaler()
    s = s.reshape((len(s), 1))
    scaler = scaler.fit(s)
    d = scaler.transform(s)
    return d

def sigmoid(x):
    a = 1 + np.exp(-x)
    return (1/a)

def sigmoid_prime(x):
    return (sigmoid(x)*(1-sigmoid(x)))

def relu(x):
    return np.maximum(x, 0)

def relu_prime(x):
    return x>0


def window_test(signal, t, window_size=200, step=1):
    data = []
    for i in range(0, len(signal) - window_size):
        tleft = 1-(i/len(signal))
        chunk = signal[i:i+window_size]
        dt = t[i:i+window_size]
        slope, intercept = np.polyfit(dt, chunk, 1)
        std = chunk.std()
        mean = chunk.mean()
        slope = slope*10**5
        std = std*10**2
        data.append((slope, intercept, std, mean, tleft))

    data = np.array(data)
    data = np.round(data, 3)
    x = data[:, :4]
    y = data[:, 4:]
    return x, y

def window_train(signal, t, window_size=200, step=1):
    data = []
    for i in range(0, len(signal) - window_size):
        tleft = 1-(i/len(signal))
        chunk = signal[i:i+window_size]
        dt = t[i:i+window_size]
        slope, intercept = np.polyfit(dt, chunk, 1)
        std = chunk.std()
        mean = chunk.mean()
        slope = slope*10**5
        std = std*10**2
        data.append((slope, intercept, std, mean, tleft))

    data = np.array(data)
    data = np.round(data, 3)
    np.random.shuffle(data)
    x = data[:, :4]
    y = data[:, 4:]
    return x, y

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

def cost(y, a2):
    # n = y.shape[1]
    # a = 0
    # for i in range(n):
    #     a += np.square(y[0, i] - a2[0, i])
    # return (a/n)
    return (y-a2)**2

def cost_prime(y, a2):
    # n = y.shape[1]
    # a = (2/n)*(a2-y)
    # return a
    return 2*(a2-y)


class network:
    def __init__(self, input_size, lr, hidden_size=4):
        self.lr = lr
        # n = x#x.shape[0]
        # m = y#x.shape[1]
        self.w1 = np.random.rand(hidden_size, input_size)
        self.b1 = np.random.rand(hidden_size, 1)
        self.w2 = np.random.rand(1, hidden_size)
        self.b2 = np.random.rand(1, 1)
    def forward(self, x):
        z1 = self.w1.dot(x) + self.b1
        a1 = relu(z1)
        z2 = self.w2.dot(a1) + self.b2
        a2 = relu(z2)
        return z1, a1, z2, a2

    def backward(self, x, y, a2, z2, a1, z1):
        c = cost(y, a2)
        dE = cost_prime(y, a2)
        da2 = dE*relu_prime(z2)
        dz2 = (self.w2.T).dot(da2)
        dw2 = da2.dot(a1.T)
        db2 = da2
        da1 = dz2*relu_prime(z1)
        dz1 = (self.w1.T).dot(da1)
        dw1 = da1.dot(x.T)
        db1 = da1
        return dw1, db1, dw2, db2, c

    def cycle(self, x, y):
        a = self.lr
        z1, a1, z2, a2 = self.forward(x)
        dw1, db1, dw2, db2, c = self.backward(x, y, a2, z2, a1, z1)
        self.w1 = self.w1 - a*dw1
        self.b1 = self.b1 - a*db1
        self.w2 = self.w2 - a*dw2
        self.b2 = self.b2 - a*db2
        return c

def init(x, y, epoch, net):
    for e in range(epoch):
        cost = np.empty(0)
        for i in range(y.size):
            X = x[i].reshape(x[i].size, 1)
            Y = y[i]
            now_cost = net.cycle(X, Y)
            cost = np.append(cost, now_cost)
        if e % 1 == 0:
            print('Epoch = ', e, 'Cost = ', np.mean(cost))

def test(x, y):
    pred = np.empty(0)
    for i in range(y.size):
        X = x[i].reshape(x[i].size, 1)
        _, _, _, o = net.forward(X)
        pred = np.append(pred, o)

    #pred = np.concatenate((np.zeros(n_steps), pred)) #add zeros on start to match original battery curve
    return pred

# path_dir = pickdir.choose_directory('data')+'/'
# path_dir = 'data/17-12-16-30-5/'
path_dir = 'data/4-2-25/'
battery = mt.battery(path_dir)
t = battery[0]
# battery = norm(battery[1])
# #power = mt.power(path_dir)
# #power = norm(power)
#
# #power = power[300:]
# battery = battery[300:]
# t = t[300:]

lr = 0.1
e = 100
input_size = 4
net = network(input_size, lr, hidden_size=1)
x_train, y_train = window_train(battery[1], battery[0])
init(x_train, y_train, e, net)
x_test, y_test = window_test(battery[1], battery[0])
pred = test(x_test, y_test)
y = y_test
plt.plot(t[:len(y)], y, label='y')
plt.plot(t[:len(y)], pred, label='pred')
plt.plot(t, battery[1], label='battery')
# #plt.plot(t, power, label='power')
plt.legend()
plt.show()


# l = 500
# t = np.linspace(0, 30, l)
# power = np.sin(t) + np.exp(t/8) + np.random.rand(l)
# power = norm(power)
# lr = 0.00001
# n = 10
# net = network(n, lr)
# init(t[:300], power[:300], n, net)
# test(t, power, n)


# train_dir = np.array(['data/17-12-16-27-41/', 'data/17-12-16-30-5/'])
# test_dir = np.array(['data/17-12-16-34-13/'])
# #path_dir = pickdir.choose_directory('data')+'/'
#
# lr = 0.00001
# n = 10
# e = 300
# net = network(n, lr)
#
# for path_dir in train_dir:
#     battery = mt.battery(path_dir)
#     t = mt.time(path_dir)
#
#     battery = norm(battery)
#     x, y = window(battery, battery, n)
#     init(x, y, n, e, net)
#
# for path_dir in test_dir:
#     battery = mt.battery(path_dir)
#     t = mt.time(path_dir)
#
#     battery = norm(battery)
#     x, y = window(battery, battery, n)
#     pred = test(x, y, n)
#     plt.plot(t, pred, label='pred')
#     plt.plot(t, battery, label='battery')
#     plt.legend()
#     plt.show()

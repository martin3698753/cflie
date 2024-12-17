import numpy as np
import maketab as mt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import maketab

def norm(s):
    scaler = MinMaxScaler()
    s = s.reshape((len(s), 1))
    scaler = scaler.fit(s)
    d = scaler.transform(s)
    return d

def relu(x):
    return np.maximum(x, 0)

def relu_prime(x):
    return x>0

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
    def __init__(self, input_size, lr):
        self.lr = lr
        hidden_size = 4
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

def init(t, power, n_steps, epoch, net):
    x, y = window(power, power, n_steps)
    for e in range(epoch):
        for i in range(y.size):
            X = x[i].reshape(x[i].size, 1)
            Y = y[i]
            cost = net.cycle(X, Y)
        if e % 50 == 0:
            print('Epoch = ', e)
            print('Cost = ', cost)

def test(t, power, n_steps):
    x, y = window(power, power, n_steps)
    pred = np.empty(0)
    for i in range(y.size):
        X = x[i].reshape(x[i].size, 1)
        _, _, _, o = net.forward(X)
        pred = np.append(pred, o)

    pred = np.concatenate((np.zeros(n_steps), pred))
    plt.plot(t, power, label='power')
    plt.plot(t, pred, label='pred')
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



path_dir = 'data/26-11-24/'
battery = mt.battery(path_dir)
#power = mt.power(path_dir)
t = mt.time(path_dir)

battery = norm(battery)
battery = battery[400:4000]
t = t[400:4000]
lr = 0.00001
n = 10
e = 200
net = network(n, lr)
init(t, battery, n, e, net)
test(t, battery, n)

# l = 300
# t = np.linspace(0, 30, l)
# #power = np.sin(t) + np.exp(t/8) + np.random.rand(l)
# power = t*3
# power = norm(power)
# power = power.reshape(-1)
# power = norm(power)
#
# n = 10
# x, y = window(power, power, n=10)
#
# x = x.T
# y = y.reshape(y.size, 1)
# y = y.T
# net = network(x, y)
# net.train(x, y)
# _,_,_,a2 = net.forward(x)
# y = y.T
# y = y.reshape(-1)
# a2 = a2.T
# a2 = a2.reshape(-1)
# a2 = np.concatenate((np.zeros(10), a2))
# #t = t[n:]
# print(a2.shape)
# print(t.shape)
# print(power.shape)
# # power = power[n:]
# # battery = battery[n:]
# plt.plot(t, a2, label='a2')
# plt.plot(t, power, label='power')
# #plt.plot(t, battery, label='battery')
# plt.legend()
# plt.show()

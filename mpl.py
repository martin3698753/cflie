import numpy as np
import maketab as mt
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(x, 0)

def relu_prime(x):
    return x > 0

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
    n = y.shape[1]
    a = 0
    for i in range(n):
        a += np.square(y[0, i] - a2[0, i])
    return (a/n)

def cost_prime(y, a2):
    n = y.shape[1]
    a = (2/n)*(a2-y)
    return a

class network:
    def __init__(self, x, y):
        n = x.shape[0]
        m = x.shape[1]
        self.w1 = np.random.rand(5, n)
        self.b1 = np.random.rand(5, m)
        self.w2 = np.random.rand(1, 5)
        self.b2 = np.random.rand(1, m)
    def forward(self, x):
        z1 = self.w1.dot(x) + self.b1
        a1 = relu(z1)
        z2 = self.w2.dot(a1) + self.b2
        a2 = relu(z2)
        return z1, a1, z2, a2

    def backward(self, x, y, a2, z2, a1, z1):
        c = cost(y, a2)
        print('Error = ', c)
        dE = cost_prime(y, a2)
        da2 = dE*relu_prime(z2)
        dz2 = (self.w2.T).dot(da2)
        dw2 = da2.dot(a1.T)
        db2 = da2
        da1 = dz2*relu_prime(z1)
        dz1 = (self.w1.T).dot(da1)
        dw1 = da1.dot(x.T)
        db1 = da1
        return dw1, db1, dw2, db2

    def train(self, x, y, e=500, a=0.00001):
        for i in range(e):
            print('Iteretion ', i)
            z1, a1, z2, a2 = self.forward(x)
            dw1, db1, dw2, db2 = self.backward(x, y, a2, z2, a1, z1)
            self.w1 = self.w1 - a*dw1
            self.b1 = self.b1 - a*db1
            self.w2 = self.w2 - a*dw2
            self.b2 = self.b2 - a*db2

path_dir = 'data/26-11-24/'
battery = mt.battery(path_dir)
power = mt.power(path_dir)
t = mt.time(path_dir)

n=20
x, y = window(power, battery, n=n)
x = x.T
y = y.reshape(y.size, 1)
y = y.T
net = network(x, y)
net.train(x, y)
_,_,_,a2 = net.forward(x)
y = y.T
y = y.reshape(-1)
a2 = a2.T
a2 = a2.reshape(-1)
t = t[n:]
power = power[n:]
battery = battery[n:]
plt.plot(t, a2, label='a2')
#plt.plot(t, y, label='y')
#plt.plot(t, y-a2, label='error')
plt.plot(t, power, label='power')
plt.plot(t, battery, label='battery')
plt.ylim(2.5,4.5)
plt.legend()
plt.show()

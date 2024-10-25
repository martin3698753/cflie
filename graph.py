import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

data = genfromtxt('data.csv', delimiter=',')

x = data[1:,0]
y = data[1:,1]
z = data[1:,2]
vbat = data[1:,3]
batlevel = data[1:,4]
t = np.arange(0, x.shape[0])
plt.plot(t, x, label='x')
plt.plot(t, y, label='y')
plt.plot(t, z, label='z')
plt.plot(t, vbat/1000-3, label='bat_voltage') #chci ukazat jenom zmenu toho
plt.plot(t, batlevel/100, label='bat_level')
plt.legend()
plt.savefig('plot.png')



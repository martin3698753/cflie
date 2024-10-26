import matplotlib.pyplot as plt
import numpy as np
import math
from numpy import genfromtxt
import pandas as pd

def sumar(ar):
    dist = 0
    s = 0
    for i in ar:
        if not math.isnan(i):
            s += i
            dist = np.append(dist, s)
    print("complete distance is",s)
    return dist

data = genfromtxt('data.csv', delimiter=',')


x = data[1:,0]
y = data[1:,1]
z = data[1:,2]
vbat = data[1:,3]
batlevel = data[1:,4]
r = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)
dist = sumar(r)
dist = np.append(dist, [0,0]) #Dummy variables

t = np.arange(0, x.shape[0])

#plt.plot(t, vbat/1000, label='vbat')
#plt.plot(t, batlevel, label='batlevel')
#plt.plot(t, dist, label='distance')
#plt.legend()
#plt.savefig('bat.png')

#plt.plot(x, y)
#plt.savefig('trace.png')

#plt.plot(t, x, label='x')
#plt.plot(t, y, label='y')
#plt.plot(t, z, label='z')
#plt.plot(t, vbat/1000-3, label='bat_voltage') #chci ukazat jenom zmenu toho
#plt.plot(t, batlevel/100, label='bat_level')
#plt.legend()
#plt.show()



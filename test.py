import lstm
import numpy as np
import maketab as mt

path_dir = 'data/26-11-24/'
# y = mt.battery(path_dir)
# t = mt.time(path_dir)
l = 300
t = np.linspace(0, 30, l)
y = np.sin(t) + np.exp(t/8) + np.random.rand(l)
lstm.init(t, y)

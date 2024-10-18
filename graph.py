import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data.csv')
t = np.arange(data.shape[0])
#data=(data-data.min())/(data.max()-data.min())
axes = list(data)
for a in axes:
    plt.plot(t, data[a], label=a)

plt.legend()
plt.savefig('nonorm.pdf')



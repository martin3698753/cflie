import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data.csv')
t = np.arange(data.shape[0])
data=(data-data.min())/(data.max()-data.min())
axes = list(data)
for a in axes:
    plt.plot(t, data[a], label=a)

#plt.plot(t, data['pm.batteryLevel'], label='level_bat')
#plt.plot(t, data['pm.vbat'], label='voltage_bat')

plt.legend()
plt.savefig('norm.png')



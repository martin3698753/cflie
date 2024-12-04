import numpy as np

def data(a):
    ws = 10 #Window size
    row = np.zeros(ws-1)
    inputs = np.zeros((0, ws-1))
    outputs = np.empty(0)

    for i in range(a.size - ws + 1):
        for j in range(ws-1):
            row[j] = a[i+j]
        inputs = np.vstack((inputs, row))
        val = a[i+ws-1]
        outputs = np.append(outputs, val)

    return inputs, outputs

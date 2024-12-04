import numpy as np

def data(a):
    ws = 5
    row = np.zeros(ws-1)

    for i in range(a.size):
        for j in range(ws-1):
            row[j] = a[i+j]
        print(row) #inputs
        print(a[i+ws-1]) #output

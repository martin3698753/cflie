import matplotlib.pyplot as plt
import numpy as np
import math
import os
import glob
import pandas as pd

# def sumar(ar):
#     dist = 0
#     s = 0
#     for i in ar:
#         if not math.isnan(i):
#             s += i
#             dist = np.append(dist, s)
#     print("complete distance is",s)
#     return dist
#
# data = genfromtxt('data.csv', delimiter=',')

if __name__ == '__main__':
    for filename in glob.glob('*.csv'):
        df = pd.read_csv(filename)
        df = df.iloc[:, :-1]
        colar = [df[col].values for col in df.columns]

        n = colar[0][0] - np.mod(colar[0][0],10)
        colar[0] = colar[0] - n

        for i in range(1,len(colar)):
            plt.plot(colar[0], colar[i], label=df.columns[i])
        plt.xlabel(df.columns[0])
        plt.legend()
        plt.title(filename)
        plt.show()

        # for column_name in df.columns:
        #     column_data = df[column_name]
        #     print(f"Column Name: {column_name}")
        #     print(column_data)
        #     print("\n" + "-"*40 + "\n")  # Separator between columns
        # t = data[1:,0]
        # x = data[1:,1]
        # y = data[1:,2]
        # z = data[1:,3]
        # n = t[0] - np.mod(t[0],10)
        # t = t - n
        # plt.plot(t, x, label='x')
        # plt.plot(t, y, label='y')
        # plt.plot(t, z, label='z')
        # plt.title(filename)
        # plt.legend()
        # plt.show()


# x = data[1:,0]
# y = data[1:,1]
# z = data[1:,2]
# deltax = data[1:,3]
# deltay = data[1:,4]
#r = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)
#dist = sumar(r)
#dist = np.append(dist, [0,0]) #Dummy variables

# t = np.arange(0, x.shape[0])

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
# plt.plot(t, deltax, label='dx')
# plt.plot(t, deltay, label='dy')
# plt.legend()
# plt.show()



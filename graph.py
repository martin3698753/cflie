import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math
import os
import glob
import pandas as pd
import pickdir
import maketab as mt
import makedata as md
import neuron


if __name__ == '__main__':
    # path_dir = pickdir.choose_directory('data')
    path_dir = 'data/26-11-24/'
    battery = mt.battery(path_dir)
    t = mt.time(path_dir)
    ins, outs = md.data(battery)
    print(ins.shape)
    print(outs.shape)

    # Assuming you have your data in X (inputs) and y (outputs)
    X = outs  # 100 samples, each with 9 features
    y = ins  # 100 target values

    # Create a QNU instance
    qnu = neuron.QNU()

    # Training loop
    num_epochs = 100
    learning_rate = 0.01

    for epoch in range(num_epochs):
        for i in range(len(X)):
            # Forward pass
            y_pred = qnu.forward(X[i].reshape(1, 9))

            # Calculate loss (e.g., mean squared error)
            loss = np.mean((y_pred - y[i])**2)

            # Calculate gradients
            dL_dout = 2 * (y_pred - y[i]) / X.shape[0]

            # Backpropagate
            qnu.backward(X[i].reshape(1, 9), dL_dout)

            # Print loss every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # position = readcsv(path_dir+'position.csv')
    # acceleration = readcsv(path_dir+'acceleration.csv')
    # t = acceleration[0]
    # power = acceleration[1]*position[1]*MASS
    # fig, axs = plt.subplots(2, layout='constrained')
    # axs[0].plot(t, power, label='power')
    # axs[0].set_xlabel('Time (s)')
    # axs[0].set_ylabel('Power (Watt)')
    # axs[0].set_title('Power consumption')
    # axs[1].plot(t, battery[1], label='batV')
    # axs[1].set_xlabel('Time (s)')
    # axs[1].set_ylabel('Voltage (V)')
    # axs[1].set_title('Battery voltage consumption')
    # plt.show()
    # pos3d(path_dir+'position.csv')

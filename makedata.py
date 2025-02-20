import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import maketab as mt
from scipy.fft import fft, fftfreq

path_dir = 'data/31-1-25/'
battery = mt.battery(path_dir)
battery = battery[:,1000:1200]
data = battery[1]
t = battery[0]

fs = 100
t = np.linspace(0, 1, fs, endpoint=False)
f1 = 50
f2 = 120
signal = 0.7*np.sin(2*np.pi*f1*t)+2*np.sin(2*np.pi*f2*t)

fft_result = fft(signal)
fft_result = np.abs(fft_result)
fft_result = np.fft.fftfreq(len(signal))
plt.plot(t, fft_result)
plt.show()
print(fft_result)
# frequencies = fftfreq(len(fft_result), 1/fs)

# # Take only the positive frequencies (since FFT output is symmetric)
# positive_frequencies = frequencies[:len(frequencies)//2]
# magnitude = np.abs(fft_result)[:len(frequencies)//2]
#
# # Find the indices of the two largest peaks
# peak_indices = np.argsort(magnitude)[-2:]  # Indices of the two largest magnitudes
# peak_frequencies = positive_frequencies[peak_indices]  # Corresponding frequencies
#
# # Print the two dominant frequencies
# print("Dominant Frequencies (Hz):", peak_frequencies)
#
# # Plot the time-domain signal
# plt.figure(figsize=(12, 6))
#
# plt.subplot(2, 1, 1)
# plt.plot(t, signal)
# plt.title('Time-Domain Signal')
# plt.xlabel('Time [s]')
# plt.ylabel('Amplitude')
# plt.grid()
#
# # Plot the frequency-domain signal
# plt.subplot(2, 1, 2)
# plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_result)[:len(frequencies)//2])
# plt.title('Frequency-Domain Signal')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Magnitude')
# plt.grid()
#
# plt.tight_layout()
# plt.show()

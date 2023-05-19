import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Generate a random pulse train
t = np.linspace(0, 1, 1000)  # Time axis
pulse_train = np.random.choice([0, 1], size=len(t), p=[0.2, 0.8])  # Random pulse train

# Design a low-pass filter
cutoff_freq = 10  # Cutoff frequency in Hz
nyquist_freq = 0.5 * 1000  # Nyquist frequency (half the sampling rate)
normalized_cutoff = cutoff_freq / nyquist_freq
b, a = signal.butter(4, normalized_cutoff, btype='low', analog=False)

# Apply the low-pass filter
filtered_signal = signal.lfilter(b, a, pulse_train)

# Plot the original pulse train and the filtered signal
plt.figure(figsize=(10, 6))
plt.plot(t, pulse_train, label='Original')
plt.plot(t, filtered_signal, label='Filtered')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Original Pulse Train and Filtered Signal')
plt.legend()
plt.show()

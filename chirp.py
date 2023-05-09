import numpy as np
from scipy import signal

# Generate a linearly chirped signal
f0 = 10  # Starting frequency
f1 = 100  # Ending frequency
T = 1  # Duration of the signal in seconds
fs = 1000  # Sampling frequency
t = np.linspace(0, T, int(T * fs), endpoint=False)  # Time vector
chirp_signal = signal.chirp(t, f0, T, f1, method='linear')

# Plot the signal
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(t, chirp_signal)
fs = 16
ax.set_xlabel('Time (s)', fontsize=fs)
ax.set_ylabel('Amplitude', fontsize=fs)
ax.set_title('Linearly Chirped Signal', fontsize=fs)
plt.show()

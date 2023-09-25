import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
b = [1, -2, 4]
a = [1, 0, 0, 0]

w, H = signal.freqz(b, 1, whole=True)
_, W = signal.freqz(a, 1, whole=True)

plt.figure()
plt.plot(w/np.pi, 20*np.log10(np.abs(H)))
plt.xlabel('Normalized Frequency (times pi rad/sample)')
plt.ylabel('Amplitude (dB)')
plt.title('Amplitude Response of Unknown Filter h[n]')
plt.grid()

plt.figure()
plt.plot(w/np.pi, np.angle(H))
plt.xlabel('Normalized Frequency (times pi rad/sample)')
plt.ylabel('Phase (rad)')
plt.title('Phase Response of Unknown Filter h[n]')
plt.grid()
plt.show()

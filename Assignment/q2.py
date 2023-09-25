import matplotlib.pyplot as plt
import numpy as np

n = np.linspace(0, 50, 51)
g1 = np.cos(2*np.pi*5*n/50) + 0.125*np.cos(2*np.pi*15*n/50)
g2 = np.cos(2*np.pi*5*n/30) + 0.125*np.cos(2*np.pi*15*n/30)
g3 = np.cos(2*np.pi*5*n/20) + 0.125*np.cos(2*np.pi*15*n/20)

fig, axs = plt.subplots(3, 1)

axs[0].stem(n, g1)
axs[0].grid(True)
axs[0].set_title('Fs = 50 Hz')

axs[1].stem(n, g2)
axs[1].grid(True)
axs[1].set_title('Fs = 30 Hz')

axs[2].stem(n, g3)
axs[2].grid(True)
axs[2].set_title('Fs = 20 Hz')

plt.show()
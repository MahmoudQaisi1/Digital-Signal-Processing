import numpy as np
import matplotlib.pyplot as plt

n = np.linspace(-10, 10, 22).astype(int)
x = np.exp((0.3j - 0.1)*n)

plt.subplot(221)
plt.stem(n, np.abs(x))
plt.xlabel('Time (s)')
plt.ylabel('Magnitude')

plt.subplot(222)
plt.stem(n, np.angle(x))
plt.xlabel('Time (s)')
plt.ylabel('Phase (rad)')

plt.subplot(223)
plt.stem(n, np.real(x))
plt.xlabel('Time (s)')
plt.ylabel('Real part')

plt.subplot(224)
plt.stem(n, np.imag(x))
plt.xlabel('Time (s)')
plt.ylabel('Imaginary part')

plt.show()
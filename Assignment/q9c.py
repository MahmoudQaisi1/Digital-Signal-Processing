import numpy as np
import matplotlib.pyplot as plt

n = np.arange(-30, 100)
x = np.zeros(len(n))
y = np.zeros(len(n))

for i in range(len(n)):
    if n[i] < 100:
        x[i] = np.sin((np.pi * n[i]) / 25)
    else:
        x[i] = 0
    if i > 0:
        y[i] = x[i] - x[i - 1]

plt.subplot(211)
plt.stem(n, x)
plt.grid()
plt.xlabel('n')
plt.ylabel('x[n]')
plt.title('x[n] = sin(2pi/25)(u[n] – u[n – 10])')

plt.subplot(212)
plt.stem(n, y)
plt.grid()
plt.xlabel('n')
plt.ylabel('y[n]')
plt.title('Simple Digital Differentiator')
plt.show()
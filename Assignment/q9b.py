import numpy as np
import matplotlib.pyplot as plt

n = np.arange(-5, 31)
x = np.zeros(len(n))
y = np.zeros(len(n))

for i in range(len(n)):
    if n[i] < 10:
        x[i] = n[i]
    elif n[i] >= 10 and n[i] < 20:
        x[i] = (20 - n[i])
    else:
        x[i] = 0
    if i > 0:
        y[i] = x[i] - x[i - 1]


plt.subplot(211)
plt.stem(n, x)
plt.grid()
plt.xlabel('n')
plt.ylabel('x[n]')
plt.title('x[n] = n (u[n] – u[n – 10]) + (20−n) (u[n – 10] – u[n – 20])')

plt.subplot(212)
plt.stem(n, y)
plt.grid()
plt.xlabel('n')
plt.ylabel('y[n]')
plt.title('Simple Digital Differentiator')
plt.show()
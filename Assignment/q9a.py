import numpy as np
import matplotlib.pyplot as plt

n = np.arange(0, 31)
x = 5*(np.heaviside(n,1) - np.heaviside(n - 20,1))
y = np.zeros(len(x))

for i in range(1,len(x)):
    y[i] = x[i]-x[i-1]


plt.subplot(211)
plt.stem(n, x)
plt.grid()
plt.xlabel('n')
plt.ylabel('x[n]')
plt.title('x[n]= 5[u(n) − u(n − 20)]')

plt.subplot(212)
plt.stem(n, y)
plt.grid()
plt.xlabel('n')
plt.ylabel('y[n]')
plt.title('Simple Digital Differentiator')
plt.show()
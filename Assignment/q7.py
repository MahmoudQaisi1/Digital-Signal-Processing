import numpy as np
import matplotlib.pyplot as plt

x = [3, 11, 7, 0, -1, 4, 2]
w = np.random.randn(len(x))
print(w)
y = [x[i - 2] + w[i] for i in range(len(x))]

r = np.correlate(y, x, mode='full')
lags = np.arange(len(r)) - len(x) + 1

y2 = [x[i - 4] + w[i] for i in range(len(x))]
r2 = np.correlate(y2, x, mode='full')
lags2 = np.arange(len(r)) - len(x) + 1

plt.subplot(211)
plt.stem(lags, r)
plt.grid()
plt.xlabel('Lags')
plt.ylabel('Cross-correlation')
plt.title('y[n]  = x[n-2] + w[n]')

plt.subplot(212)
plt.stem(lags2, r2)
plt.grid()
plt.xlabel('Lags')
plt.ylabel('Cross-correlation')
plt.title('y[n]  = x[n-4] + w[n]')
plt.show()
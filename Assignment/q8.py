import numpy as np
from scipy.signal import lfilter
import matplotlib.pyplot as plt

a = [1, -1, 0.9]
b = [1]
n = np.arange(-20,121)

x = np.concatenate((np.zeros(20), [1], np.zeros(120)))
h = lfilter(b, a, x)

x2 = np.concatenate((np.zeros(20), np.ones(121)))
s = lfilter(b, a, x2)

plt.figure()
plt.subplot(2,1,1)
plt.stem(n, h)
plt.grid()
plt.title('impulse response')

plt.subplot(2,1,2)
plt.stem(n, s)
plt.grid()
plt.title('step response')
plt.show()

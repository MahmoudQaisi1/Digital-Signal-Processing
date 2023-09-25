import numpy as np
import matplotlib.pyplot as plt

n = np.linspace(0, 50, 50).astype(int)

y = np.cos(0.04*np.pi*n) + 0.2 * (np.random.randn(50))

plt.stem(n, y)
plt.xlabel('n')
plt.ylabel('y')
plt.title('Cosine function + Gaussian random sequence')
plt.show()
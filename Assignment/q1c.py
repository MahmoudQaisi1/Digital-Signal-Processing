import numpy as np
import matplotlib.pyplot as plt

n = np.linspace(-10, 9, 20).astype(int)
z = np.array([5,4,3,2,1,5,4,3,2,1,5,4,3,2,1])
z = np.hstack([np.zeros((n.shape[0] - z.shape[0], )), z])

plt.stem(n, z)
plt.xlabel('n')
plt.ylabel('y')
plt.show()
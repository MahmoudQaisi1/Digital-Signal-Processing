import numpy as np
import matplotlib.pyplot as plt

n = np.linspace(-5,45).astype(int)
x = np.zeros(n.shape)
x[np.argwhere(n == 0)[0][0]:np.argwhere(n == 10)[0][0]] = 1

h = np.zeros(n.shape)
h[np.argwhere(n == 0)[0][0]:] = 1
h = h* 0.9**n

y = np.convolve(x,h)
n2 = np.linspace(-5,45+len(h)-1,len(y)).astype(int)
figure, axis = plt.subplots(3, 1)

axis[0].stem(n, x)
axis[0].set_title("X")

axis[1].stem(n, h)
axis[1].set_title("H")

axis[2].stem(n2, y)
axis[2].set_title("Y")

plt.show()
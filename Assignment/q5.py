import numpy as np
import matplotlib.pyplot as plt

nx = np.linspace(-3, 3, 7).astype(int)
x = np.array([3, 11, 7, 0 , -1, 4, 2])

nh = np.linspace(-1, 4, 6).astype(int)
h = np.array([2, 3, 0, -5, 2, 1])

ny = np.linspace(nx[0] + nh[0], nx[len(nx)-1] + nh[len(nh)-1]).astype(int)
ny = np.unique(ny)
y = np.convolve(x,h)
figure, axis = plt.subplots(3, 1)

axis[0].stem(nx, x)
axis[0].set_title("X")

axis[1].stem(nh, h)
axis[1].set_title("H")

axis[2].stem(ny, y)
axis[2].set_title("Y")

plt.show()
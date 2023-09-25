import matplotlib.pyplot as plt
import numpy as np


# Define the impulse function
def impulse(t, d):
    return 1 if t - d == 0 else 0


n = np.linspace(-5, 5, 12).astype(int)

y = [2 * impulse(i, -2) + impulse(i, 4) for i in n]

# Plot the impulse function
plt.stem(n, y)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Impulse Function')
plt.show()

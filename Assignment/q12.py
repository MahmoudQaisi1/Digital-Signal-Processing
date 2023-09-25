import numpy as np
import matplotlib.pyplot as plt

# Define L and the frequency range
L = 100
w = np.linspace(-2*np.pi, 2*np.pi, 401)

# Define the signal x[n]
n = np.arange(0, L+1)
x = np.cos(np.pi*n/2)

# Compute the DTFT of x[n]
X = (1/L)*np.fft.fft(x, len(w))

# Compute the signal y[n]
y = np.exp(1j*np.pi*n/4)*x

# Compute the DTFT of y[n]
Y = (1/L)*np.fft.fft(y, len(w))

# Plot the magnitude of x[n]
plt.subplot(2,2,1)
plt.plot(w, np.abs(X))
plt.grid()
plt.xlabel('w')
plt.ylabel('Magnitude')
plt.title('Magnitude of x[n]')

# Plot the phase of x[n]
plt.subplot(2,2,2)
plt.plot(w, (180/np.pi)*np.angle(X))
plt.grid()
plt.xlabel('w')
plt.ylabel('Phase')

plt.subplot(2,2,3)
plt.plot(w, np.abs(Y))
plt.grid()
plt.xlabel('w')
plt.ylabel('Magnitude')
plt.title('Magnitude Spectrum of y[n]')

plt.subplot(2,2,4)
plt.plot(w, np.angle(Y))
plt.grid()
plt.xlabel('w')
plt.ylabel('Phase')
plt.title('Angle Spectrum of y[n]')

plt.show()
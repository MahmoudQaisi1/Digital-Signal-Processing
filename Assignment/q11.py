import numpy as np
import matplotlib.pyplot as plt

x = [1, -0.5, -0.3, -0.1]

X = np.fft.fft(x, 501)

w = np.linspace(0, np.pi, 501)

plt.figure()
plt.subplot(2,2,1)
plt.plot(w, np.abs(X))
plt.xlabel('w')
plt.ylabel('|X(e^jw)|')
plt.title('Magnitude')

plt.subplot(2,2,2)
plt.plot(w, np.angle(X))
plt.xlabel('w')
plt.ylabel('arg(X(e^jw))')
plt.title('Angle')

plt.subplot(2,2,3)
plt.plot(w, np.real(X))
plt.xlabel('w')
plt.ylabel('Re(X(e^jw))')
plt.title('Real part')

plt.subplot(2,2,4)
plt.plot(w, np.imag(X))
plt.xlabel('w')
plt.ylabel('Im(X(e^jw))')
plt.title('Imaginary part')

plt.show()
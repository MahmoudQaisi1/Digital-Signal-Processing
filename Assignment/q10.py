import numpy as np
import matplotlib.pyplot as plt

w = np.linspace(0, np.pi, 501)
x = np.exp(1j*w)/(np.exp(1j*w) - 0.5)

plt.figure()
plt.subplot(2,2,1)
plt.plot(w, np.abs(x))
plt.xlabel('w')
plt.ylabel('|x(e^jw)|')
plt.title('Magnitude')

plt.subplot(2,2,2)
plt.plot(w, np.angle(x))
plt.xlabel('w')
plt.ylabel('arg(x(e^jw))')
plt.title('Angle')

plt.subplot(2,2,3)
plt.plot(w, np.real(x))
plt.xlabel('w')
plt.ylabel('Re(x(e^jw))')
plt.title('Real part')

plt.subplot(2,2,4)
plt.plot(w, np.imag(x))
plt.xlabel('w')
plt.ylabel('Im(x(e^jw))')
plt.title('Imaginary part')

plt.show()

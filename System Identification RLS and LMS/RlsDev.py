import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# part1
# a- Generate and plot the input x[n]
N = 2000
Xn = np.cos(0.03 * np.pi * np.arange(N))
Dn = Xn - (2 * np.roll(Xn, 1)) + (4 * np.roll(Xn, 2))
b = [1, -2, 4]

wd, Hd = signal.freqz(b, 1, whole=True)
Hx = np.copy(Hd)
wx = np.copy(wd)

def rls2(x, d, M, mu=0.1, lam=0.98):
    N = len(x)
    w = np.zeros(M)
    P = np.eye(M) / mu
    y = np.zeros(N)
    e = np.zeros(N)
    for n in range(M, N):
        x1 = x[n - M + 1:n + 1][::-1]
        y[n] = np.dot(w, x1)
        e[n] = d[n] - y[n]
        k = np.dot(P, x1) / (lam + np.dot(np.dot(x1, P), x1))
        w = w + k * e[n]
        P = (P - np.dot(k[:, np.newaxis], x1[np.newaxis, :]) * lam) / lam
    J = e ** 2

    Jnew = 10 * np.log10(J)
    return w, y, e,J,Jnew
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
w, y, e, J, Jnew = rls2(Xn, Dn, 4)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
ax1.plot(e)
ax1.set_title("Error")
w1, H = signal.freqz(w, 1, whole=True)
plt.subplot(2, 1, 2)
plt.plot(w1 / np.pi, np.angle(Hx), label="d")
plt.plot(w1 / np.pi, np.angle(H), label="y")
plt.xlabel("Frequency (rad/sample)")
plt.ylabel("Phase (rad)")
plt.title("Phase Response")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
"""

ax2.plot(w1 / np.pi, np.angle(H))
ax2.set_title('Phase Response of FIR Filter h[n]')

ax2.plot(wx/np.pi, np.angle(Hx))
ax2.set_title('Phase Response of Unknown Filter h[n]')
ax2.legend()
plt.tight_layout()
plt.show()
"""

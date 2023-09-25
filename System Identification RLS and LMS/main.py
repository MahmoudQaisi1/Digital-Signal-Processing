import matplotlib.pyplot as plt
import numpy as np

def lms1(x, dn, M=4, mu=0.01):
    N = len(x) #2000
    w = np.zeros(M)
    w1 = np.zeros((N-M, M))
    y = np.zeros(N)
    e = np.zeros(N)
    for n in range(M, N):
        x1 = x[n-M+1:n+1][::-1]
        y[n] = np.dot(w, x1)
        e[n] = dn[n] - y[n]
        w = w + 2 * mu * e[n] * x1
        w1[n-M] = w
    J = e**2
    Js = np.zeros(N-5)
    for n in range(0, N-5):
        Js[n] = (J[n] + J[n+1] + J[n+2]) / 3
    Jnew = 10 * np.log10(J)
    return w, y, e, J, w1, Js,Jnew

def rls(x, dn, order, forgetting_factor=1.0):
    N = len(x)
    w = np.zeros(order)
    P = np.eye(order) / forgetting_factor
    y_n = np.zeros(N)
    e_n = np.zeros(N)
    J_n = np.zeros(N)
    for n in range(order, N):
        x_n = x[n-order:n][::-1]
        y_n[n] = np.dot(w, x_n)
        e_n[n] = dn[n] - y_n[n]
        J_n[n] = e_n[n]**2
        k_n = np.dot(P, x_n) / (1 + np.dot(np.dot(x_n, P), x_n))
        w = w + k_n * e_n[n]
        P = P / forgetting_factor - np.dot(np.dot(k_n, x_n), P) / forgetting_factor
    return w, y_n, e_n, J_n

N=2000
Xn=np.cos(0.03 * np.pi * np.arange(N).astype(int))
Xf = np.fft.fft(Xn)
f = np.fft.fftfreq(N, d=1.0/N)
Xspectrum = np.abs(Xf)
plt.plot(f, Xspectrum)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude Spectrum")
plt.show()
plt.close()
plt.stem(Xn,"y")
#plt.show()
plt.close()
#print(Xn)
def generate_dn(x):
    N = len(x)
    dn = np.zeros(N)
    for n in range(2, N):
        dn[n] = x[n] - 2 * x[n-1] + 4 * x[n-2]
    return dn
Dn=generate_dn(Xn)
dn_f = np.fft.fft(Dn)
f = np.fft.fftfreq(N, d=1.0/N)
magnitude_response = np.abs(dn_f)
phase_response = np.angle(dn_f)

plt.subplot(2, 1, 1)
plt.plot(f, magnitude_response)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude Response")

plt.subplot(2, 1, 2)
plt.plot(f, phase_response)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase Response")

plt.tight_layout()
plt.show()
w, y, e, J, w1, Js, Jnew = lms1(Xn,Dn) #w = coeff, y=y[n], e=error (diffrence between Y and the adaptive filter), J= error squared
#w, y, e_n, J_n = rls(Xn,Dn,4)
plt.title("Adaptation")
plt.xlabel("Iteration n")
#plt.stem(y, "p")
#plt.stem(Dn, "y")
print(w,w1)
for i in range(2):
    plt.plot([wi[i] for wi in w1], label=f"w{i}")
#plt.plot(w1[0],"r")
#plt.plot(w1[1],"r")
#plt.plot(w1[2],"r")
#plt.plot(w1[3],"r")
plt.legend()
#plt.show()
plt.close()
plt.stem(J,"r")
#plt.show()
plt.close()


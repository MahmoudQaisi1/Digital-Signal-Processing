import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# part1
# a- Generate and plot the input x[n]
N = 2000
Xn = np.cos(0.03 * np.pi * np.arange(N).astype(int))
'''
plt.stem(Xn,"y")
plt.show()
plt.close()
'''
Dn = Xn - (2 * np.roll(Xn, 1)) + (4 * np.roll(Xn, 2))
'''
# b- Plot the amplitude and phase response for the given FIR system.

b = [1, -2, 4]

w, H = signal.freqz(b, 1, whole=True)

plt.figure()
plt.plot(w/np.pi, np.log10(np.abs(H)))
plt.xlabel('Normalized Frequency (times pi rad/sample)')
plt.ylabel('Amplitude (dB)')
plt.title('Amplitude Response of Unknown Filter h[n]')
plt.grid()

plt.figure()
plt.plot(w/np.pi, np.angle(H))
plt.xlabel('Normalized Frequency (times pi rad/sample)')
plt.ylabel('Phase (rad)')
plt.title('Phase Response of Unknown Filter h[n]')
plt.grid()
plt.show()
'''

'''
# C- Plot the spectrum for the input signal x[n].
Xf = np.fft.fft(Xn)
f = np.fft.fftfreq(N, d=1.0 / N)
Xspectrum = np.abs(Xf)
plt.stem(f, Xspectrum)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude Spectrum")
plt.show()
plt.close()
'''


# D- Implement the LMS algorithm to estimate the filter coefficients w0,…. w3
# μ is assumed to be very small (try μ =0.01).

def rls(x, dn, order, forgetting_factor=0.8):
    N = len(x)
    w = np.zeros(order)
    P = np.eye(order) / forgetting_factor
    y = np.zeros(N)
    e = np.zeros(N)
    J = np.zeros(N)
    for n in range(order, N):
        x_n = x[n - order:n][::-1]
        y[n] = np.dot(w, x_n)
        e[n] = dn[n] - y[n]
        J[n] = e[n] ** 2
        k_n = np.dot(P, x_n) / (1 + np.dot(np.dot(x_n, P), x_n))
        w = w + k_n * e[n]
        P = P / forgetting_factor - np.dot(np.dot(k_n, x_n), P) / forgetting_factor
    Jnew = 10 * np.log10(J)
    return w, y, e, J, w, Jnew


import numpy as np


import numpy as np

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


import numpy as np

def rls1(x, dn, lam=1, M=4):
    N = len(x)  # 2000
    w = np.zeros(M)
    w1 = np.zeros((N - M, M))
    y = np.zeros(N)
    e = np.zeros(N)
    P = np.eye(M) / lam
    for n in range(M, N):
        x1 = x[n - M + 1:n + 1][::-1]
        y[n] = np.dot(w, x1)
        e[n] = dn[n] - y[n]
        k = np.dot(P, x1) / (1 + np.dot(x1, np.dot(P, x1)))
        w = w + k * e[n]
        P = (P - np.dot(k[:, np.newaxis], x1[np.newaxis, :]) * np.dot(x1, P)) / lam
        w1[n - M] = w
    J = e ** 2

    Jnew = 10 * np.log10(J)

    return w, y, e, J, w1, Jnew


#w, y, e, J, w1, Jnew = rls(Xn,Dn,4)
#w, y, e, J, w1, Jnew = rls1(Xn,Dn,0.8,4)
w,y,e,J,Jnew=rls2(Xn,Dn,4)


# 1- error learning rate e(n):

plt.plot(e)
plt.show()

# 2- j = e(n)^2

plt.plot(J)
plt.show()

# 3- 10log10j:

plt.plot(Jnew)
plt.show()


# Plot the amplitude and phase response for the estimated FIR system at the end of the iterations.
# Compare it with the given FIR system.

w1, H = signal.freqz(w, 1, whole=True)

plt.figure()
plt.plot(w1/np.pi, np.log10(np.abs(H)))
plt.xlabel('Normalized Frequency (times pi rad/sample)')
plt.ylabel('Amplitude (dB)')
plt.title('Amplitude Response of Unknown Filter h[n]')
plt.grid()

plt.figure()
plt.plot(w1/np.pi, np.angle(H))
plt.xlabel('Normalized Frequency (times pi rad/sample)')
plt.ylabel('Phase (rad)')
plt.title('Phase Response of Unknown Filter h[n]')
plt.grid()
plt.show()

# F- Decrease the value of μ.
# What is the effect of changing μ on the speed of the learning process and on the steady state error?

plt.stem(Dn, "y")
plt.stem(y, "r")
plt.legend()
plt.show()
plt.plot(e, "y")
w, y, e, J, w1, Jnew = rls(Xn,Dn,4,0.1)
plt.plot(e, "k")
plt.legend()
plt.show()


#Add 40dB of zeros mean white Gaussian noise to x[n]
# (hint: use awgn() mathlab function). Repeat parts (D)-(F).
# Give your conclusions.
'''
noise = np.random.normal(0, np.sqrt(10**(-40/10)), N)
Xn_noisy = Xn + noise
Dn = Xn - (2 * np.roll(Xn_noisy, 1)) + (4 * np.roll(Xn_noisy, 2))
w, y, e, J, w1, Jnew = rls(Xn_noisy,Dn,4)

# 1- error learning rate e(n):

plt.plot(e)
plt.show()

# 2- j = e(n)^2

plt.plot(J)
plt.show()

# 3- 10log10j:

plt.plot(Jnew)
plt.show()

plt.plot(y, "k")
plt.plot(Dn, "y")
plt.legend()
plt.show()

plt.plot(e, "k")
w, y, e, J, w1, Jnew = rls(Xn_noisy, Dn, 4)
plt.plot(e, "y")
plt.legend()
plt.show()
Xn_noisy
'''
'''
#Repeat part (G) for 30dB. Try to modify the step size value.
noise = np.random.normal(0, np.sqrt(10**(-30/10)), N)
Xn_noisy = Xn + noise
Dn = Xn - (2 * np.roll(Xn_noisy, 1)) + (4 * np.roll(Xn_noisy, 2))
w, y, e, J, w1, Jnew = rls(Xn_noisy, Dn,4)

# 1- error learning rate e(n):

plt.plot(e)
plt.show()

# 2- j = e(n)^2

plt.plot(J)
plt.show()

# 3- 10log10j:

plt.plot(Jnew)
plt.show()

plt.plot(y, "k")
plt.plot(Dn, "y")
plt.legend()
plt.show()

plt.plot(e, "k")
w, y, e, J, w1, Jnew = rls(Xn_noisy, Dn, 4)
plt.plot(e, "y")
plt.legend()
plt.show()
'''
'''
#Ensemble averaging:
#Repeat part (G) for 1000 trials and average the obtained J over the number of trials.
#plot the averaged J (10log10(J) vs iteration steps)

noise = np.random.normal(0, np.sqrt(10**(-40/10)), N)
Xn_noisy = Xn + noise
Dn = Xn - (2 * np.roll(Xn_noisy, 1)) + (4 * np.roll(Xn_noisy, 2))

num_trials = 1000
J_sum = np.zeros(N)
for i in range(num_trials):
    _, _, _, J, _, _ = rls(Xn_noisy, Dn, 4)
    J_sum = J_sum + J

J_avg = J_sum / num_trials

plt.plot(J_avg)
plt.xlabel("Sample Number")
plt.ylabel("Squared Error (dB)")
plt.title("Averaged J Over 1000 Trials")
plt.show()
'''
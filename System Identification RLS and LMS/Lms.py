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

# b- Plot the amplitude and phase response for the given FIR system.

b = [1, -2, 4]

wd, Hd = signal.freqz(b, 1, whole=True)

plt.figure()
plt.plot(wd/np.pi, np.log10(np.abs(Hd)))
plt.xlabel('Normalized Frequency (times pi rad/sample)')
plt.ylabel('Amplitude (dB)')
plt.title('Amplitude Response of Unknown Filter h[n]')
plt.grid()

plt.figure()
plt.plot(wd/np.pi, np.angle(Hd))
plt.xlabel('Normalized Frequency (times pi rad/sample)')
plt.ylabel('Phase (rad)')
plt.title('Phase Response of Unknown Filter h[n]')
plt.grid()
plt.show()


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

def lms1(x, dn, mu=0.005, M=4):
    N = len(x)  # 2000
    w = np.zeros(M)
    w1 = np.zeros((N - M, M))
    y = np.zeros(N)
    e = np.zeros(N)
    for n in range(M, N):
        x1 = x[n - M + 1:n + 1][::-1]
        y[n] = np.dot(w, x1)
        e[n] = dn[n] - y[n]
        w = w + 2 * mu * e[n] * x1
        w1[n - M] = w
    J = e ** 2

    Jnew = 10 * np.log10(J)

    return w, y, e, J, w1, Jnew


w, y, e, J, w1, Jnew = lms1(Xn, Dn)


# 1- error learning rate e(n):

# Plot error, error squared, and 10log10 error

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

ax1.plot(e)
ax1.set_title("Error")

ax2.plot(J)
ax2.set_title("Error Squared")

ax3.plot(Jnew)
ax3.set_title("10log10 Error")

plt.tight_layout()
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
plt.show()
plt.figure()
plt.plot(w1/np.pi, np.angle(H))
plt.xlabel('Normalized Frequency (times pi rad/sample)')
plt.ylabel('Phase (rad)')
plt.title('Phase Response of Unknown Filter h[n]')
plt.grid()
plt.show()

# F- Decrease the value of μ.
# What is the effect of changing μ on the speed of the learning process and on the steady state error?
'''
plt.plot(e, "y")
w, y, e, J, w1, Jnew = rls(Xn, Dn, mu=0.005)
plt.plot(e, "k")
plt.legend()
plt.show()
'''

#Add 40dB of zeros mean white Gaussian noise to x[n]
# (hint: use awgn() mathlab function). Repeat parts (D)-(F).
# Give your conclusions.
'''
noise = np.random.normal(0, np.sqrt(10**(-40/10)), N)
Xn_noisy = Xn + noise
Dn = Xn - (2 * np.roll(Xn_noisy, 1)) + (4 * np.roll(Xn_noisy, 2))
w, y, e, J, w1, Jnew = rls(Xn_noisy, Dn)

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
w, y, e, J, w1, Jnew = rls(Xn_noisy, Dn, mu=0.005)
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
w, y, e, J, w1, Jnew = rls(Xn_noisy, Dn)

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
w, y, e, J, w1, Jnew = rls(Xn_noisy, Dn, mu=0.005)
plt.plot(e, "y")
plt.legend()
plt.show()
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
    _, _, _, J, _, _ = lms1(Xn_noisy, Dn)
    J_sum = J_sum + J

J_avg = J_sum / num_trials

plt.plot(J_avg)
plt.xlabel("Sample Number")
plt.ylabel("Squared Error (dB)")
plt.title("Averaged J Over 1000 Trials")
plt.show()
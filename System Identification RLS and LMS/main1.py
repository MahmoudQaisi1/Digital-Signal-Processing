import matplotlib.pyplot as plt
import numpy as np
#from scipy import signal

# part1
# a- Generate and plot the input x[n]
N = 2000
Xn = np.cos(0.03 * np.pi * np.arange(N).astype(int))
'''
plt.stem(Xn,"y")
plt.show()
plt.close()
'''


# b- Plot the amplitude and phase response for the given FIR system.
def generate_dn(x):
    N = len(x)
    dn = np.zeros(N)
    for n in range(2, N):
        dn[n] = x[n] - 2 * x[n - 1] + 4 * x[n - 2]
    return dn


Dn = generate_dn(Xn)
print(Dn,len(Dn))
t=np.arange(2000)
Xn1 = np.cos(0.03 * np.pi * t)
Xn2= -2 * np.cos(0.03 * np.pi * t-1)
Xn3= 4 * np.cos(0.03* np.pi * t-2)
Dn=Xn1+Xn2+Xn3
print(Dn,len(Dn))
# Plotting the Amplitude Response
w, h = np.abs(np.fft.fft(Dn)), np.angle(np.fft.fft(Dn))
freq = np.fft.fftfreq(Dn.size, d=1./N)
#w_d, h_d = freqz(d_n)
#plt.plot(w, 20 * np.log10(np.abs(h_d)), label="d_n")

plt.figure()
plt.subplot(2,1,1)
plt.plot(freq, w)
plt.title('Amplitude Response')
plt.xlabel('Frequency [rad/sample]')
plt.ylabel('Magnitude')

# Plotting the Phase Response
plt.subplot(2,1,2)
plt.plot(freq, h)
plt.title('Phase Response')
plt.xlabel('Frequency [rad/sample]')
plt.ylabel('Phase [rad]')

plt.tight_layout()
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

def lms1(x, dn, mu=0.01, M=4):
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
    Js = np.zeros(N - 5)
    for n in range(0, N - 5):
        Js[n] = (J[n] + J[n + 1] + J[n + 2]) / 3

    Jnew = 10 * np.log10(J)

    return w, y, e, J, w1, Js, Jnew


w, y, e, J, w1, Js, Jnew = lms1(Xn, Dn)
'''
# 1- error learning rate e(n):

plt.plot(e)
plt.show()

# 2- j = e(n)^2

plt.plot(J)
plt.show()

# 3- 10log10j:

plt.plot(Jnew)
plt.show()
'''
# Plot the amplitude and phase response for the estimated FIR system at the end of the iterations.
# Compare it with the given FIR system.
'''
plt.plot(y, "g")
plt.show()
plt.close()
plt.plot(Dn, "y")
plt.plot(y, "k")
plt.legend()
plt.show()
plt.close()

# Plotting the Amplitude Response
w, h = np.abs(np.fft.fft(y)), np.angle(np.fft.fft(y))
freq = np.fft.fftfreq(y.size, d=1./N)

plt.figure()
plt.subplot(2,1,1)
plt.plot(freq, abs(h))
plt.title('Amplitude Response')
plt.xlabel('Frequency [rad/sample]')
plt.ylabel('Magnitude')

# Plotting the Phase Response
plt.subplot(2,1,2)
plt.plot(freq, np.angle(h))
plt.title('Phase Response')
plt.xlabel('Frequency [rad/sample]')
plt.ylabel('Phase [rad]')

plt.tight_layout()
plt.show()
'''

# F- Decrease the value of μ.
# What is the effect of changing μ on the speed of the learning process and on the steady state error?
'''
#plt.stem(e, "p")
mu=0.005
w, y, e, J, w1, Js, Jnew = lms1(Xn, Dn, mu=mu)
plt.subplot(2,1,1)
plt.plot(J, "y")
plt.title(f"error squared with u={mu}")
plt.subplot(2,1,2)
plt.plot(e, "y")
plt.title( f"error with u={mu}")
plt.tight_layout()
plt.show()
plt.plot(Dn, "y")
plt.plot(y, "k")
plt.legend()
plt.show()
'''

'''
#Add 40dB of zeros mean white Gaussian noise to x[n]
# (hint: use awgn() mathlab function). Repeat parts (D)-(F).
# Give your conclusions.

noise = np.random.normal(0, np.sqrt(10**(-40/10)), N)
Xn_noisy = Xn + noise
Dn = generate_dn(Xn_noisy)
w, y, e, J, w1, Js, Jnew = lms1(Xn_noisy, Dn)

# 1- error learning rate e(n):

plt.plot(e)
plt.show()

# 2- j = e(n)^2

plt.plot(J)
plt.show()

# 3- 10log10j:

plt.plot(Jnew)
plt.show()

plt.plot(Dn, "y")
plt.plot(y, "k")
plt.legend()
plt.show()

plt.plot(e, "g")
w, y, e, J, w1, Js, Jnew = lms1(Xn_noisy, Dn, mu=0.005)
plt.plot(e, "y")
plt.legend()
plt.show()
'''

'''
#Repeat part (G) for 30dB. Try to modify the step size value.
noise = np.random.normal(0, np.sqrt(10**(-30/10)), N)
Xn_noisy = Xn + noise
Dn = generate_dn(Xn_noisy)
w, y, e, J, w1, Js, Jnew = lms1(Xn_noisy, Dn)

# 1- error learning rate e(n):

plt.stem(e)
plt.show()

# 2- j = e(n)^2

plt.stem(J)
plt.show()

# 3- 10log10j:

plt.stem(Jnew)
plt.show()

plt.stem(y, "p")
plt.stem(Dn, "y")
plt.legend()
plt.show()

plt.stem(e, "p")
w, y, e, J, w1, Js, Jnew = lms1(Xn_noisy, Dn, mu=0.005)
plt.stem(e, "y")
plt.legend()
plt.show()
'''

#Ensemble averaging:
#Repeat part (G) for 1000 trials and average the obtained J over the number of trials.
#plot the averaged J (10log10(J) vs iteration steps)
'''
noise = np.random.normal(0, np.sqrt(10**(-40/10)), N)
Xn_noisy = Xn + noise
Dn = generate_dn(Xn_noisy)

num_trials = 1000
J_sum = np.zeros(N)
for i in range(num_trials):
    _, _, _, J, _, _, _ = lms1(Xn_noisy, Dn)
    J_sum = J_sum + J

J_avg = J_sum / num_trials

plt.plot(J_avg)
plt.xlabel("Sample Number")
plt.ylabel("Squared Error (dB)")
plt.title("Averaged J Over 1000 Trials")
plt.show()
'''
#!/usr/bin/env python
# coding: utf-8

# In[1]:


#EYE DIAGRAM

fimport numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift

def get_filter(name, T, rolloff=None):
    def rc(t, beta):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.sinc(t)*np.cos(np.pi*beta*t)/(1-(2*beta*t)**2)
    def rrc(t, beta):
        return (np.sin(np.pi*t*(1-beta))+4*beta*t*np.cos(np.pi*t*(1+beta)))/(np.pi*t*(1-(4*beta*t)**2))


    if name == 'rc':
        return lambda t: rc(t/T, rolloff)
    elif name == 'rrc':
        return lambda t: rrc(t/T, rolloff)

T = 1

Fs = 100
t = np.arange(-3*T, 3*T, 1/Fs)
g = get_filter('rc', T, rolloff=0.4)
alpha=0.5

plt.figure(figsize=(8,3))
plt.plot(t, get_filter('rc', T, rolloff=0.4)(t), 'b--', label=r'Raised cosine $\alpha=0.4$')
plt.plot(t, get_filter('rrc', T, rolloff=0.4)(t), 'g--', label=r'Root raised cosine $\alpha=0.4$')
plt.legend()
b = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0])
d = 2*b-1
print ("d=%s" % d)

def get_signal(g, d):
    """Generate the transmit signal as sum(d[k]*g(t-kT))"""
    t = np.arange(-2*T, (len(d)+2)*T, 1/Fs)
    g0 = g(np.array([1e-8]))
    xt = sum(d[k]*g(t-k*T) for k in range(len(d)))
    return t, xt/g0

fig = plt.figure(figsize=(8,3))
t, xt = get_signal(g, d)
plt.plot(t, xt, 'k-', label='$x(t)$')
plt.legend()
plt.stem(T*np.arange(len(d)), d,'r--',label='data')
plt.legend()
for k in range(len(d)):
    plt.plot(t, d[k]*g(t-k*T), 'b--', label='$d[k]g(t-kT)$')

def drawFullEyeDiagram(xt):
    samples_perT = Fs*T
    samples_perWindow = 2*Fs*T
    parts = []
    startInd = 2*samples_perT

    for k in range(int(len(xt)/samples_perT) - 6):
        parts.append(xt[startInd + k*samples_perT + np.arange(samples_perWindow)])
    parts = np.array(parts).T

    t_part = np.arange(-T, T, 1/Fs)
    plt.plot(t_part, parts, 'b-')

def drawSignals(g, data=None):

    N = 100;
    if data is None:
        data = 2*((np.random.randn(N)>0))-1
       
        data[0:10] = 2*np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 0])-1

    t, xt = get_signal(g, data)

    plt.subplot(223)
    t_g = np.arange(-4*T, 4*T, 1/Fs)
    plt.plot(t_g, g(t_g))

    plt.subplot(211)
    plt.plot(t, xt)
    plt.stem(data)

    plt.subplot(224)
    drawFullEyeDiagram(xt); plt.ylim((-2,2))
    plt.tight_layout()

d = 2*np.array([0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1])-1
print ("d=%s" % d)
plt.figure(figsize=(8,6))
N = 100
data = 3-2*np.arange(4)[np.random.randint(4, size=N)]
print (data[:10])
drawSignals(get_filter('rc', T=1, rolloff=0.4), data=data)


# In[2]:


#Error performance of BPSK
import numpy as np # for numerical computing
import matplotlib.pyplot as plt # for plotting functions
from scipy.special import erfc

def bpsk_mod(ak, L):
    from scipy.signal import upfirdn
    s_bb = upfirdn(h=[1]*L, x=2*ak-1, up=L) # NRZ encoder
    t = np.arange(start=0, stop=len(ak)*L) # discrete time base
    return (s_bb, t)

def bpsk_demod(r_bb, L):
    x = np.real(r_bb) # I arm
    x = np.convolve(x, np.ones(L)) # integrate for \( T_b \) duration (L samples)
    x = x[L-1::L] # I arm - sample at every L
    ak_hat = (x > 0).transpose() # threshold detector
    return ak_hat

from numpy import sum, isrealobj, sqrt
from numpy.random import standard_normal

def awgn(s, SNRdB, L=1):
    gamma = 10**(SNRdB/10) # SNR to linear scale
    if s.ndim == 1: # if \( s \) is single dimensional vector
        P = L * sum(abs(s)**2)/len(s) # Actual power in the vector
    else: # multi-dimensional signals like MFSK
        P = L * sum(sum(abs(s)**2))/len(s) # if \( s \) is a matrix [M×N]
    N0 = P/gamma # Find the noise spectral density
    if isrealobj(s): # check if input is real/complex object type
        n = sqrt(N0/2)*standard_normal(s.shape) # computed noise
    else:
        n = sqrt(N0/2)*(standard_normal(s.shape) + 1j*standard_normal(s.shape))
    r = s + n # received signal
    return r
N = 100000 # Number of symbols to transmit
EbN0dB = np.arange(start=-4, stop=11, step=2) # \( E_b/N_0 \) range in dB for simulation
L = 16 # oversampling factor, \( L = T_b/T_s \) (\( T_b \): bit period, \( T_s \): sampling period)
# if a carrier is used, use \( L = F_s/F_c \), where \( F_s \gg 2×F_c \)

Fc = 800 # carrier frequency
Fs = L*Fc # sampling frequency
BER = np.zeros(len(EbN0dB)) # for BER values for each \( E_b/N_0 \)

ak = np.random.randint(2, size=N) # uniform random symbols from 0's and 1's
(s_bb, t) = bpsk_mod(ak, L) # BPSK modulation (waveform) - baseband
s = s_bb * np.cos(2*np.pi*Fc*t/Fs) # with carrier

# Waveforms at the transmitter
fig1, axs = plt.subplots(2, 2)
axs[0, 0].plot(t, s_bb) # baseband waveform zoomed to first 10 bits
axs[0, 0].set_xlabel("t(s)"); axs[0, 1].set_ylabel(r'$s_{bb}(t)$ - baseband')
axs[0, 1].plot(t, s) # transmitted waveform zoomed to first 10 bits
axs[0, 1].set_xlabel("t(s)"); axs[0, 1].set_ylabel(r's(t) - with carrier')
axs[0, 0].set_xlim(0, 10*L); axs[0, 1].set_xlim(0, 10*L)

# Signal constellation at transmitter
axs[1, 0].plot(np.real(s_bb), np.imag(s_bb), 'o')
axs[1, 0].set_xlim(-1.5, 1.5); axs[1, 0].set_ylim(-1.5, 1.5)
for i, EbN0 in enumerate(EbN0dB):
    # Compute and add AWGN noise
    r = awgn(s, EbN0, L)
    r_bb = r * np.cos(2*np.pi*Fc*t/Fs) # recovered baseband signal
    ak_hat = bpsk_demod(r_bb, L) # baseband correlation demodulator
    BER[i] = np.sum(ak != ak_hat)/N # Bit Error Rate Computation

    # Received signal waveform zoomed to first 10 bits
    axs[1, 1].plot(t, r) # received signal (with noise)
    axs[1, 1].set_xlabel("t(s)"); axs[1, 1].set_ylabel(r'r(t)')
    axs[1, 1].set_xlim(0, 10*L)

# ----- Theoretical Bit/Symbol Error Rates --------------
theoreticalBER = 0.5 * erfc(np.sqrt(10**(EbN0dB/10))) # Theoretical bit error rate

# ----------- Plots --------------------
fig2, ax1 = plt.subplots(nrows=1, ncols=1)
ax1.semilogy(EbN0dB, BER, 'k*', label='Simulated') # simulated BER
ax1.semilogy(EbN0dB, theoreticalBER, 'r-', label='Theoretical') # theoretical BER
ax1.set_xlabel(r'$E_b/N_0$ (dB)')
ax1.set_ylabel(r'Probability of Bit Error - $P_b$')
ax1.set_title(['Probability of Bit Error for BPSK Modulation'])
ax1.legend()
fig1.show()
fig2.show()



# In[3]:


#Error performance of QPSK
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import upfirdn
from scipy.special import erfc
from numpy import sum, isrealobj, sqrt
from numpy.random import standard_normal

def qpsk_mod(a, fc, OF, enable_plot=True):
    L = 2 * OF
    I = a[0::2]
    Q = a[1::2]
    I = upfirdn(h=[1] * L, x=2 * I - 1, up=L)
    Q = upfirdn(h=[1] * L, x=2 * Q - 1, up=L)
    fs = OF * fc
    t = np.arange(0, len(I) / fs, 1 / fs)
    I_t = I * np.cos(2 * np.pi * fc * t)
    Q_t = -Q * np.sin(2 * np.pi * fc * t)
    s_t = I_t + Q_t
    if enable_plot:
        fig = plt.figure(constrained_layout=True)
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(3, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[-1, :])
        ax1.plot(t, I)
        ax2.plot(t, Q)
        ax3.plot(t, I_t, "r")
        ax4.plot(t, Q_t, "r")
        ax1.set_title("I(t)")
        ax2.set_title("Q(t)")
        ax3.set_title(r"$I(t) cos(2 \pi f_c t)$")
        ax4.set_title(r"$Q(t) sin(2 \pi f_c t)$")
        ax1.set_xlim(0, 20 * L / fs)
        ax2.set_xlim(0, 20 * L / fs)
        ax3.set_xlim(0, 20 * L / fs)
        ax4.set_xlim(0, 20 * L / fs)
        ax5.plot(t, s_t)
        ax5.set_xlim(0, 20 * L / fs)
        ax5.set_title(r"$s(t) = I(t) cos(2 \pi f_c t) - Q(t) sin(2 \pi f_c t)$")
        fig, axp = plt.subplots(1, 1)
        axp.plot(np.real(I), np.real(Q), "o")
    result = dict()
    result["s(t)"] = s_t
    result["I(t)"] = I
    result["Q(t)"] = Q
    result["t"] = t
    return result

def qpsk_demod(r, fc, OF, enable_plot=True):
    fs = OF * fc
    L = 2 * OF
    t = np.arange(0, len(r) / fs, 1 / fs)
    x = r * np.cos(2 * np.pi * fc * t)
    y = -r * np.sin(2 * np.pi * fc * t)
    x = np.convolve(x, np.ones(L))
    y = np.convolve(y, np.ones(L))
    x = x[L - 1 :: L]
    y = y[L - 1 :: L]
    a_hat = np.zeros(2 * len(x))
    a_hat[0::2] = (x > 0)
    a_hat[1::2] = (y > 0)
    if enable_plot:
        fig, axs = plt.subplots(1, 1)
        axs.plot(x[0:200], y[0:200], "o")
    return a_hat

def awgn(s, SNRdB, L=1):
    gamma = 10 ** (SNRdB / 10)
    if s.ndim == 1:
        P = L * sum(abs(s) ** 2) / len(s)
    else:
        P = L * sum(sum(abs(s) ** 2)) / len(s)
    N0 = P / gamma
    if isrealobj(s):
        n = sqrt(N0 / 2) * standard_normal(s.shape)
    else:
        n = sqrt(N0 / 2) * (
            standard_normal(s.shape) + 1j * standard_normal(s.shape)
        )
    r = s + n
    return r

N = 100000
EbN0dB = np.arange(start=-4, stop=11, step=2)
fc = 100
OF = 8
BER = np.zeros(len(EbN0dB))
a = np.random.randint(2, size=N)
result = qpsk_mod(a, fc, OF, enable_plot=True)
s = result["s(t)"]

for i, EbN0 in enumerate(EbN0dB):
    r = awgn(s, EbN0, OF)
    a_hat = qpsk_demod(r, fc, OF)
    BER[i] = np.sum(a != a_hat) / N

theoreticalBER = 0.5 * erfc(np.sqrt(10 ** (EbN0dB / 10)))
fig, axs = plt.subplots(nrows=1, ncols=1)
axs.semilogy(EbN0dB, BER, "k*", label="Simulated")
axs.semilogy(EbN0dB, theoreticalBER, "r-", label="Theoretical")
axs.set_title("Probability of Bit Error for QPSK modulation")
axs.set_xlabel(r"$E_b/N_0$ (dB)")
axs.set_ylabel(r"Probability of Bit Error - $P_b$")
axs.legend()
plt.show()


# In[ ]:





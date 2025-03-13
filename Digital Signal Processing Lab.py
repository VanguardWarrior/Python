#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from scipy.signal import freqz

X=13

#frequency components
(f1, f2, f3)=(X, 2*X, 2*X+0.1)
#sampling frequencies
sampling_freqs=[4*X, 4.5*X, 10*X, 2.4*X]
N=32
N2=64

def signal_gen(fs):
    """function to generate signal having 3 frequency components"""

    t=np.arange(0,1,1/fs)
    signal=np.zeros_like(t)
    signal=np.sin(2*np.pi*f1*t)+np.sin(2*np.pi*f2*t)+np.sin(2*np.pi*f3*t)
    return signal

#generate and plot 32 point dft(fft)

plotn=1

for fs in sampling_freqs:

    signal=signal_gen(fs)
    dft_32=fft.fft(signal[:N],N)
    freqs_32= np.fft.fftfreq(N, 1/fs)
    plt.figure(figsize=(20,20))
    plt.subplot(12,1, plotn)
    plt.stem(freqs_32, np.abs(dft_32))
    plt.xticks(np.arange(0, max(freqs_32),8))
    plt.xlim([0,max(freqs_32)])
    plt.title(f"32 point DFT with sampling frequency {fs}Hz")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plotn+=1

#generate and plot 64 point dft (fft)

for fs in sampling_freqs:

    signal=signal_gen(fs)
    dft_64= fft.fft(np.concatenate([signal [:N], np.zeros(N)]), N2)
    freqs_64= np.fft.fftfreq (N2, 1/fs)
    plt.figure(figsize=(20,20))
    plt.subplot(12,1, plotn)
    plt.stem(freqs_64, np.abs(dft_64))
    plt.xticks(np.arange(0,max(freqs_64),8))
    plt.xlim([0,max(freqs_64)])
    plt.title(f"64 point DFT with sampling frequency {fs}Hz")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plotn+=1

#generate and plot DTFT

for fs in sampling_freqs:

    signal=signal_gen(fs)
    w,h=freqz(signal)
    freqs_dtft=w*fs /(2*np.pi)
    plt.figure(figsize=(20,20))
    plt.subplot(12,1,plotn)
    plt.plot(freqs_dtft,np.abs(h))
    plt.xticks(np.arange(0, max(freqs_dtft),8))
    plt.xlim([0,max(freqs_dtft)])
    plt.title(f"DTFT with sampling frequency {fs}Hz")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plotn+=1


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

def bpsk_mod(ak, L):
    from scipy.signal import upfirdn
    s_bb = upfirdn(h=[1]*L, x=2*ak-1, up=L)
    t = np.arange(start=0, stop=len(ak)*L)
    return (s_bb, t)

def bpsk_demod(r_bb, L):
    x = np.real(r_bb)
    x = np.convolve(x, np.ones(L))
    x = x[L-1::L]
    ak_hat = (x > 0).transpose()
    return ak_hat

from numpy import sum, isrealobj, sqrt
from numpy.random import standard_normal

def awgn(s, SNRdB, L=1):
    gamma = 10**(SNRdB/10)
    if s.ndim == 1:
        P = L * sum(abs(s)**2)/len(s)
    else:
        P = L * sum(sum(abs(s)**2))/len(s)
    N0 = P/gamma
    if isrealobj(s):
        n = sqrt(N0/2)*standard_normal(s.shape)
    else:
        n = sqrt(N0/2)*(standard_normal(s.shape) + 1j*standard_normal(s.shape))
    r = s + n
    return r
N = 100000
EbN0dB = np.arange(start=-4, stop=11, step=2)
L = 16

Fc = 800
Fs = L*Fc
BER = np.zeros(len(EbN0dB))

ak = np.random.randint(2, size=N)
(s_bb, t) = bpsk_mod(ak, L)
s = s_bb * np.cos(2*np.pi*Fc*t/Fs)

fig1, axs = plt.subplots(2, 2)
axs[0, 0].plot(t, s_bb)
axs[0, 0].set_xlabel("t(s)"); axs[0, 1].set_ylabel(r'$s_{bb}(t)$ - baseband')
axs[0, 1].plot(t, s)
axs[0, 1].set_xlabel("t(s)"); axs[0, 1].set_ylabel(r's(t) - with carrier')
axs[0, 0].set_xlim(0, 10*L); axs[0, 1].set_xlim(0, 10*L)

axs[1, 0].plot(np.real(s_bb), np.imag(s_bb), 'o')
axs[1, 0].set_xlim(-1.5, 1.5); axs[1, 0].set_ylim(-1.5, 1.5)
for i, EbN0 in enumerate(EbN0dB):
    
    r = awgn(s, EbN0, L)
    r_bb = r * np.cos(2*np.pi*Fc*t/Fs)
    ak_hat = bpsk_demod(r_bb, L)
    BER[i] = np.sum(ak != ak_hat)/N

    axs[1, 1].plot(t, r)
    axs[1, 1].set_xlabel("t(s)"); axs[1, 1].set_ylabel(r'r(t)')
    axs[1, 1].set_xlim(0, 10*L)

theoreticalBER = 0.5 * erfc(np.sqrt(10**(EbN0dB/10)))

fig2, ax1 = plt.subplots(nrows=1, ncols=1)
ax1.semilogy(EbN0dB, BER, 'k*', label='Simulated')
ax1.semilogy(EbN0dB, theoreticalBER, 'r-', label='Theoretical')
ax1.set_xlabel(r'$E_b/N_0$ (dB)')
ax1.set_ylabel(r'Probability of Bit Error - $P_b$')
ax1.set_title(['Probability of Bit Error for BPSK Modulation'])
ax1.legend()
fig1.show()
fig2.show()


# In[2]:


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
from IPython.core.display import display, HTML

# Disable scrollbars for outputs
display(HTML("<style>div.output_scroll { height: auto; }</style>"))

plt.show()


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
n=np.arange(-5,6)
#x=np.zeros_like(n)
x[(n>=-3)&(n<0)]=-1
x[(n>=0)&(n<3)]=1
plt.stem(n,x)
plt.xlabel("n")
plt.ylabel("x[n]")
plt.xticks(n)


# In[2]:


a = np.arange(6) 
b = np.arange(4, -1, -1) 
x = np.concatenate([a, b])
plt.stem(x) 
plt.xticks(np.arange(11));


# In[3]:


import numpy as np
N=4
X=np.array([[1,2,3,4]]).T
D=np.zeros_like(X,dtype=np.cdouble)
for k in np.arange(N):
    for n in np.arange(N):
        D[k]=D[k]+X[n]*np.exp(-1j*np.pi*2*n*k/N)

np.round(D)
print(D)


# In[4]:


import numpy as np
N=4
D=np.empty((N,N),dtype=np.cdouble)
w=np.exp(-1j*np.pi*2/N)
for k in np.arange(N):
    for n in np.arange(N):
        D[k,n]=w**(k*n)
X=np.array([[1,2,3,4]]).T
x=D*X
np.round(x)
print(D)


# In[5]:


import numpy as np
x=np.arange(5)
X=np.zeros_like(x,dtype=np.cdouble)
N=4
for k in range(4):
    for n in range(4):
        X[k]=X[k]+x[n]*np.exp(-1j*2*np.pi*k*n/N)
dft=X.round()
print(dft)


# In[7]:


import numpy as np
def circon(x,h):
    if len(x)>len(h):
        N=len(x)
        for i in range(len(x)-len(h)):
            h=np.append(h,0)
    if len(h)>len(x):
        N=len(h)
        for i in range(len(h)-len(x)):
            x=np.append(x,0)
    if len(x)==len(h):
        N=len(h)
    conv=np.zeros(N)
    for n in range(N):
        for k in range(N):
            conv[n]+=x[k]*h[(n-k)%N]
    print(conv)
    return conv
x=np.array([1,2,3,4,5])
h=np.array([2,2,0,1,1])
circon(x,h)


# In[8]:


import numpy as np
import matplotlib.pyplot as plt
n=np.arange(0,11)
x=np.zeros_like(n)
x[n]=n
plt.stem(n,x)


# In[9]:


import numpy as np
import matplotlib.pyplot as plt
def func(n):
    return (0.95**n)*np.cos(0.1*np.pi*n)
n=np.arange(0,51)
plt.stem(n,func(n))


# In[10]:


import numpy as np
import matplotlib.pyplot as plt
def corrupt_sig(sig,sdev):
    noise=np.random.normal(0,sdev,sig.size)
    return noise+sig
sdev=[0,0.2,0.7,1.3]
t= np.arange(0,2,0.01)
sig=np.sin(2*np.pi*t)
for i in range(len(sdev)):
    signal=corrupt_sig(sig,sdev[i])
    plt.figure(figsize=(5,10))
    plt.subplot(len(sdev),1,i+1)
    plt.plot(t,signal,label="Signal with gaussian noise of "+str(sdev[i]))
    plt.legend()
    plt.tight_layout()


# In[11]:


import numpy as np
import matplotlib.pyplot as plt
def corrupt_sig(sig,sdev):
    noise=np.random.normal(0,sdev,sig.size)
    return noise+sig
sdev=[0,0.2,0.7,1.3]
t= np.arange(0,2,0.01)
sig=np.sin(2*np.pi*t)
for i in range(len(sdev)):
    signal=corrupt_sig(sig,sdev[i])
    plt.figure(figsize=(5,10))
    plt.subplot(len(sdev),1,i+1)
    plt.plot(t,signal,label="Signal with gaussian noise of "+str(sdev[i]))
    plt.legend()
    plt.tight_layout()


# In[12]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import dimpulse, freqz

a = [0, 9, -2, 1]
b = [1, -1]

system = (b, a)
t, h = dimpulse(system, n=20)
w, h_freq = freqz(b, a, worN=8000)
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.stem(t, np.squeeze(h), basefmt=" ", use_line_collection=True)
plt.title('Impulse Response')
plt.xlabel('n (samples)')
plt.ylabel('Amplitude')


plt.subplot(2, 1, 2)
plt.plot(w / np.pi, 20 * np.log10(abs(h_freq)))
plt.title('Frequency Response')
plt.xlabel('Normalized Frequency (×π rad/sample)')
plt.ylabel('Magnitude (dB)')
plt.grid()

plt.tight_layout()
plt.show()


# In[13]:


def linear_convolution(x, h):
    
    len_x = len(x)
    len_h = len(h)
   
    len_y = len_x + len_h - 1
    y = [0] * len_y 
    for i in range(len_y):
        for j in range(len_h):
            if i - j >= 0 and i - j < len_x:
                y[i] += x[i - j] * h[j]
    return y

x = [1,2,3,4]
h = [1,2]

result = linear_convolution(x, h)
print("Linear Convolution Result:", result)


# In[14]:


import numpy as np
import matplotlib.pyplot as plt

# Define parameters
f1 = 20   # Frequency of the first sine wave (20 Hz)
f2 = 50   # Frequency of the second sine wave (50 Hz)
fs = 200  # Sampling frequency (200 Hz)
N = 20    # Number of samples

# Generate time vector for N samples at fs sampling frequency
t = np.arange(N) / fs

# Generate the sine waves
sin_wave1 = np.sin(2 * np.pi * f1 * t)
sin_wave2 = np.sin(2 * np.pi * f2 * t)

# Combine the sine waves
combined_signal = sin_wave1 + sin_wave2

# Compute the DFT of the combined signal
dft_result = np.fft.fft(combined_signal)

# Compute the frequencies corresponding to the DFT result
frequencies = np.fft.fftfreq(N, 1/fs)

# Plotting the individual sine waves, combined signal, and DFT magnitude
plt.figure(figsize=(14, 8))

# Plot the first sine wave
plt.subplot(3, 1, 1)
plt.plot(t, sin_wave1, label='20 Hz Sine Wave')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('20 Hz Sine Wave')
plt.grid()
plt.legend()

# Plot the second sine wave
plt.subplot(3, 1, 2)
plt.plot(t, sin_wave2, label='50 Hz Sine Wave')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('50 Hz Sine Wave')
plt.grid()
plt.legend()

# Plot the combined signal
plt.subplot(3, 1, 3)
plt.plot(t, combined_signal, label='Combined Signal (20 Hz + 50 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Combined Signal (20 Hz + 50 Hz)')
plt.grid()
plt.legend()

# Show the combined sine waves
plt.tight_layout()
plt.show()

# Plot the magnitude of the DFT result
plt.figure(figsize=(8, 4))
plt.stem(frequencies[:N//2], np.abs(dft_result)[:N//2], 'b', markerfmt=" ", basefmt="-b")
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.title('DFT Magnitude of the Combined Signal')
plt.grid()
plt.show()


# In[15]:


import numpy as np
import matplotlib.pyplot as plt
x=[1,2,3]
N=3
nk=np.arange(N)
xk=np.zeros_like(x)
for k in range(N):
    for n in range(N):
        xk[k]=xk[k]+x[n]*np.exp(-1j*2*np.pi*k*n/N)
magxk=np.abs(xk)
plt.figure(figsize=(5,10))
plt.subplot(3,1,1)
plt.stem(nk,x)
plt.title("input sequence")
plt.subplot(3,1,2)
plt.stem(nk,magxk)
plt.title("DFT sequence")
xn=np.zeros_like(x)
for n in range(N):
    for k in range(N):
        xn[n]=xn[n]+xk[k]*np.exp(1j*2*np.pi*k*n/N)
magxn=np.abs(xn)
plt.subplot(3,1,3)
plt.stem(nk,magxn)
plt.title("IDFT sequence")
plt.tight_layout()
plt.show()


# In[16]:


import numpy as np

x = [1, 2, 3]
N = 3
X=np.zeros_like(x,dtype=np.cdouble)
y=np.zeros_like(x,dtype=np.cdouble)
for k in range(N):
    for n in range(N):
        X[k]=X[k]+x[n]*np.exp(-1j * 2* np.pi * k * n / N)
print("input sequence ",x)
print("dft=",np.abs(X))
for n in range(N):
    for k in range(N):
        y[n]=y[n]+(X[k]*np.exp(1j * 2* np.pi * k * n / N))/N
print("idft ",np.abs(y))


# In[17]:


import numpy as np
import matplotlib.pyplot as plt
def corrupt_sig(sig,sdev):
    noise=np.random.normal(0,sdev,sig.size)
    return sig+noise
N=100
f=50
fs=200
sdev=0.3
n=np.arange(N)
t=n/fs
sig=np.sin(2*np.pi*f*t)
noisy_sig=corrupt_sig(sig,sdev)
dft_noise=np.fft.fft(noisy_sig)
plt.figure(figsize=(8,15))
plt.subplot(3,1,1)
plt.stem(t,sig)
plt.title("original signal")
plt.subplot(3,1,2)
plt.stem(t,noisy_sig)
plt.title("noisy signal")
w=n*fs/N
plt.subplot(3,1,3)
plt.stem(w,np.abs(dft_noise))
plt.title("dft")
plt.show()


# In[18]:


import numpy as np
import matplotlib.pyplot as plt

x=[1,2,3,4]
h=[1,2,3,4]
lx=len(x)
lh=len(h)
n=lx+lh-1

for i in range(n-lx):
    x=np.append(x,0)
for i in range(n-lh):
    h=np.append(h,0)
h=np.flip(h)
a=h[n-1]
y=np.zeros_like(x,dtype=np.cdouble)

for j in range(n,-1,1):
    h[j]=h[j-1]
h[0]=a
for i in range(n):
    y[i]=0
    for j in range(n):
        y[i]=y[i]+(x[j]*h[j])
    a=h[n-1]
    for k in range(n,-1,1):
        h[j]=h[j-1]
    h[0]=a
plt.stem(y)
print(y)


# In[19]:


x = [1, 2, 3, 4]
h = [1, 3, 5]

N = max(len(x), len(h))

x = x + [0] * (N - len(x))
h = h + [0] * (N - len(h))

y = [0] * N

for n in range(N):
    for m in range(N):
        y[n] += x[m] * h[(n - m) % N]

print("Circular Convolution Result:", y)


# In[20]:


import numpy as np
import matplotlib.pyplot as plt

f1=50
f2=60
fs=200
N=20
n=np.arange(N)
t=n/fs

sig1=np.sin(2*np.pi*f1*t)
sig2=np.sin(2*np.pi*f2*t)

comb_sig=sig1+sig2

lhs=np.sum(np.abs(comb_sig**2))
rhs=(1/N)*np.sum(np.abs(np.fft.fft(comb_sig))**2)

print(lhs,rhs)
plt.subplot(2,1,1)
plt.stem(n,comb_sig)
w=n*fs/N
plt.subplot(2,1,2)
plt.stem(w,np.abs(np.fft.fft(comb_sig)))
if(np.round(lhs)==np.round(rhs)):
    print("parsevals theorem is verified")
else:
    print("parsevals theorem is not verified")


# In[21]:


import numpy as np
x=[2,4,3,6]
l=len(x)
y=[0]*l
w=[0]*l

for k in range(l):
    y[k]=0
    if(k==0):
        w[k]=1/np.sqrt(l)
    else:
        w[k]=np.sqrt(2/l)
    for n in range(l):
        y[k]=y[k]+x[n]*np.cos(np.pi*(2*n-1)*(k-1)/(2*l))
    y[k]=y[k]*w[k]
mag=np.abs(y)
print(np.round(mag))
phase=np.angle(y)
print(np.round(phase))


# In[22]:


import numpy as np
import matplotlib.pyplot as plt

# Input sequence
x = np.array([float(i) for i in input("Enter the sequence (comma-separated): ").split(',')])
l = len(x)  # length of the sequence

# DCT calculation
y = np.zeros(l)
w = np.zeros(l)
for k in range(l):
    y[k] = 0
    if k == 0:
        w[k] = 1 / np.sqrt(l)
    else:
        w[k] = np.sqrt(2 / l)
    for n in range(l):
        y[k] += x[n] * np.cos(np.pi * (2 * n + 1) * k / (2 * l))
    y[k] *= w[k]

# Plotting input sequence
t = np.arange(l)
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.stem(t, x, use_line_collection=True)
plt.xlabel("n ---->")
plt.ylabel("Amplitude ---->")
plt.title("Input Sequence")
plt.grid(True)

# DCT magnitude and phase
magnitude = np.abs(y)
phase = np.angle(y)

# Display DCT results
print("DCT Sequence = ")
print(magnitude)
print("Phase = ")
print(phase)

# Plot DCT magnitude
plt.subplot(2, 2, 2)
plt.stem(t, magnitude, use_line_collection=True)
plt.xlabel("K ---->")
plt.ylabel("Amplitude ---->")
plt.title("DCT Sequence")
plt.grid(True)

# Plot phase response
plt.subplot(2, 2, 3)
plt.stem(t, phase, use_line_collection=True)
plt.xlabel("K ---->")
plt.ylabel("Phase ---->")
plt.title("Phase Response")
plt.grid(True)

# IDCT calculation
X = np.zeros(l)
for n in range(l):
    X[n] = 0
    if n == 0:
        w[n] = 1 / np.sqrt(l)
    else:
        w[n] = np.sqrt(2 / l)
    for k in range(l):
        X[n] += w[k] * y[k] * np.cos(np.pi * (2 * n + 1) * k / (2 * l))

# Display IDCT result
print("IDCT Sequence = ")
print(X)

# Plot IDCT sequence
plt.subplot(2, 2, 4)
plt.stem(t, X, use_line_collection=True)
plt.xlabel("n ---->")
plt.ylabel("Amplitude ---->")
plt.title("IDCT Sequence")
plt.grid(True)

plt.tight_layout()
plt.show()


# In[23]:


import numpy as np
import matplotlib.pyplot as plt

N=20
f=30
fs=100
n=np.linspace(0,fs/f,N)
t=n/fs
sig=np.sin(2*np.pi*f*t)
plt.plot(n,sig)


# In[24]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from scipy.signal import freqz

X=13

#frequency components
(f1, f2, f3)=(X, 2*X, 2*X+0.1)
#sampling frequencies
sampling_freqs=[4*X, 4.5*X, 10*X, 2.4*X]
N=32
N2=64

def signal_gen(fs):
    """function to generate signal having 3 frequency components"""

    t=np.arange(0,1,1/fs)
    signal=np.zeros_like(t)
    signal=np.sin(2*np.pi*f1*t)+np.sin(2*np.pi*f2*t)+np.sin(2*np.pi*f3*t)
    return signal

#generate and plot 32 point dft(fft)

plotn=1

for fs in sampling_freqs:

    signal=signal_gen(fs)
    dft_32=fft.fft(signal[:N],N)
    freqs_32= np.fft.fftfreq(N, 1/fs)
    plt.figure(figsize=(20,20))
    plt.subplot(12,1, plotn)
    plt.stem(freqs_32, np.abs(dft_32))
    plt.xticks(np.arange(0, max(freqs_32),8))
    plt.xlim([0,max(freqs_32)])
    plt.title(f"32 point DFT with sampling frequency {fs}Hz")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plotn+=1

#generate and plot 64 point dft (fft)

for fs in sampling_freqs:

    signal=signal_gen(fs)
    dft_64= fft.fft(np.concatenate([signal [:N], np.zeros(N)]), N2)
    freqs_64= np.fft.fftfreq (N2, 1/fs)
    plt.figure(figsize=(20,20))
    plt.subplot(12,1, plotn)
    plt.stem(freqs_64, np.abs(dft_64))
    plt.xticks(np.arange(0,max(freqs_64),8))
    plt.xlim([0,max(freqs_64)])
    plt.title(f"64 point DFT with sampling frequency {fs}Hz")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plotn+=1

#generate and plot DTFT

for fs in sampling_freqs:

    signal=signal_gen(fs)
    w,h=freqz(signal)
    freqs_dtft=w*fs /(2*np.pi)
    plt.figure(figsize=(20,20))
    plt.subplot(12,1,plotn)
    plt.plot(freqs_dtft,np.abs(h))
    plt.xticks(np.arange(0, max(freqs_dtft),8))
    plt.xlim([0,max(freqs_dtft)])
    plt.title(f"DTFT with sampling frequency {fs}Hz")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plotn+=1
    


# In[ ]:





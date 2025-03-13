#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
t=np.arange(0,10,0.01)
a=np.sin(t)
b=np.cos(t)
c=np.sinh(t)
d=np.cosh(t)

fig=plt.figure(figsize=(15,10))
plt.subplots_adjust(wspace=0.4,hspace=0.4)
            
plt.subplot(2,2,1)
plt.plot(t,a)
plt.title('Sine wave')
plt.xlabel("Time")
plt.ylabel("Amplitude")


plt.subplot(2,2,2)
plt.plot(t,b)
plt.title('Cosine wave')
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.subplot(2,2,3)
plt.plot(t,c)
plt.title('Hyperbolic sine wave',)
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.subplot(2,2,4)
plt.plot(t,d)
plt.title('Hyperbolic cosine wave')
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.show()


# In[12]:


import numpy as np
import matplotlib.pyplot as plt
t=np.arange(0,10,0.01)
a=np.sin(t)
b=np.cos(t)
c=np.sinh(t)
d=np.cosh(t)

def der(m,n):
    d1=np.gradient(m,n)
    d2=np.gradient(d1,n)
    return d1,d2
d1a,d2a=der(a,0.01)
d1b,d2b=der(b,0.01)
d1c,d2c=der(c,0.01)
d1d,d2d=der(d,0.01)

plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
plt.plot(t,a)
plt.plot(t,d1a)
plt.plot(t,d2a)
plt.title('Sine and derivatives')
plt.xlabel('Values of t')
plt.ylabel('Amplitude')
plt.legend(["Sine","First derivative","Second derivative"],loc="lower right")
plt.grid()

plt.subplot(2,2,2)
plt.plot(t,b)
plt.plot(t,d1b)
plt.plot(t,d2b)
plt.title('Cos and derivatives')
plt.xlabel('Values of t')
plt.ylabel('Amplitude')
plt.legend(["Cosine","First derivative","Second derivative"],loc="lower right")
plt.grid()

plt.subplot(2,2,3)
plt.plot(t,c)
plt.plot(t,d1c)
plt.plot(t,d2c)
plt.title('Hyperbolic sine and derivatives')
plt.xlabel('Values of t')
plt.ylabel('Amplitude')
plt.legend(["Hyperbolic sine","First derivative","Second derivative"],loc="upper left")
plt.grid()

plt.subplot(2,2,4)
plt.plot(t,d)
plt.plot(t,d1d)
plt.plot(t,d2d)
plt.title('Hyperbolic cosine and derivatives')
plt.xlabel('Values of t')
plt.ylabel('Amplitude')
plt.legend(["Hyperbolic cosine","First derivative","Second derivative"],loc="upper left")
plt.grid()


# In[16]:


from numpy import exp,sqrt,pi,inf,arange
from scipy.integrate import simps,trapz,quad
def f(x):
    return (1/sqrt(2*pi))*exp((-x**2)/2)
res,err=quad(f,0,float(inf))
interval=arange(f,0,float(inf))
func=f(interval)
traps=trapz(func,interval)
simpson=simps(func,interval)
print(res)
print(traps)
print(simpson)


# In[ ]:





# In[21]:


from numpy import exp,sqrt,pi,inf,arange
from scipy.integrate import simps,trapz,quad
def f(x):
    return (1/sqrt(2*pi))*exp((-x**2)/2)
res,err=quad(f,0,float(inf))
interval=arange(0,40000)
func=f(interval)
traps=trapz(func,interval)
simpson=simps(func,interval)
print(res)
print(traps)
print(simpson)


# In[1]:


import random
N=[100,500,1000,5000,500000]
head_probarr=[]
for i in (N):
    hcount=0
    for j in range(i):
        toss=random.randint(0,1)
        if(toss==1):
            hcount+=1
    phead=hcount/i
    head_probarr.append(phead)
print("Probability of Heads:",head_probarr)


# In[2]:


import random
import matplotlib.pyplot as plt
N=[100,500,1000,5000,500000]
Head_proberror=[]
Tail_proberror=[]
for i in (N):
    hcount=0
    tcount=0
    for j in range(i):
        toss=random.randint(0,1)
        if(toss==1):
            hcount+=1
        else:
            tcount+=1
    phead=hcount/i
    ptail=tcount/i
    head_abserr=abs(0.5-phead)
    tail_abserr=abs(0.5-ptail)
    Head_proberror.append(head_abserr)
    Tail_proberror.append(tail_abserr)
print("Probability error in heads:",Head_proberror)
print("Probability error in tails:",Tail_proberror)

plt.plot(N,Head_proberror)
plt.title("Probability error in heads")
plt.show()

plt.plot(N,Tail_proberror)
plt.title("Probability error in tails")
plt.show()
    


# In[3]:


def squares(a):
    for i in range(len(a)):
        a[i]=a[i]**2
    return a
a=[3,2,8,4,5]
print(squares(a))
print(len(a))


# In[4]:


from numpy import sin,pi,arange,linspace
import matplotlib.pyplot as plt

k=linspace(1,10,100)
y=arange(-2*pi,2*pi,0.01)
plt.plot(y,sin(y*pi))
#plt.stem(sin(k*pi))
plt.show()


# In[5]:


from numpy import real,imag,abs
arr=([1+3j,2-4j,5+9j])
imgArr=[]
for i in range(3):
    im=imag(arr[i])
    imgArr.append(im)
print(imgArr)


# In[6]:


import numpy as np
import matplotlib.pyplot as plt
x=np.arange(0,10,1)
y=sin(x)

plt.subplot(2,1,1)
plt.stem(x,y,linefmt='r',markerfmt='k')
plt.title("stem plot")

plt.subplot(2,1,2)
plt.plot(x,y,'g')
plt.title("continuos plot")
plt.tight_layout()
plt.show()


# In[7]:


import numpy as np
import matplotlib.pyplot as plt
t=np.arange(0,10,0.01)
a=np.sin(t)
b=np.cos(t)
c=np.sinh(t)
d=np.cosh(t)

plt.subplots_adjust(wspace=0.4,hspace=0.4)
fig=plt.figure(figsize=(15,30))

plt.subplot(4,1,1)
plt.plot(t,a)

plt.subplot(4,1,2)
plt.plot(t,b)

plt.subplot(4,1,3)
plt.plot(t,c)

plt.subplot(4,1,4)
plt.plot(t,d)
plt.show()


# In[8]:


from numpy import sin,cos,sinh,cosh,arange,gradient
import matplotlib.pyplot as plt

t=arange(0,10,0.01)
a=sin(t)
b=cos(t)
c=sinh(t)
d=cosh(t)

def der(m,n):
    d1=gradient(m,n)
    d2=gradient(d1,n)
    return d1,d2
d1a,d2a=der(a,0.01)
d1b,d2b=der(b,0.01)
d1c,d2c=der(c,0.01)
d1d,d2d=der(d,0.01)

fig=plt.figure(figsize=(15,10))
plt.subplots_adjust(wspace=0.4,hspace=0.4)

plt.subplot(2,2,1)
plt.grid()
plt.plot(t,a)
plt.plot(t,d1a)
plt.plot(t,d2a)
plt.title("sine")
plt.xlabel("t")
plt.ylabel("amplitude")
plt.legend(["sine","first derivative","second derivative"],loc="upper right")

plt.subplot(2,2,2)
plt.grid()
plt.plot(t,b)
plt.plot(t,d1b)
plt.plot(t,d2b)
plt.title("cosine")
plt.xlabel("t")
plt.ylabel("amplitude")
plt.legend(["cosine","first derivative","second derivative"],loc="upper right")


plt.subplot(2,2,3)
plt.grid()
plt.plot(t,c)
plt.plot(t,d1c)
plt.plot(t,d2c)
plt.title("sineh")
plt.xlabel("t")
plt.ylabel("amplitude")
plt.legend(["sinh","first derivative","second derivative"],loc="lower left")


plt.subplot(2,2,4)
plt.grid()
plt.plot(t,d)
plt.plot(t,d1d)
plt.plot(t,d2d)
plt.title("cosh")
plt.xlabel("t")
plt.ylabel("amplitude")
plt.legend(["cosh","first derivative","second derivative"],loc="lower left")
plt.show()


# In[9]:


from scipy.integrate import quad
def f(t):
    return 1
res,err=quad(f,0,2)
print(res)


# In[10]:


from numpy import real,imag,abs
x=3+5j
print("The real no:",real(x))
print("The im no:",imag(x))
print("the absolute value of x:",abs(x))


# In[11]:


import numpy as np
import matplotlib.pyplot as plt
time=np.linspace(-2*np.pi,2*np.pi,100)
amplitude=np.sinc(time)
plt.plot(time,amplitude,color='r')
plt.title("sinc function",color='y')
plt.grid()
plt.show()


# In[12]:


from numpy import sqrt,sin,arange
from math import pi
x=arange(0,1.001*pi,0.01*pi)
print(sum(sqrt(x)*sin(x)))


# In[13]:


import numpy as np
import matplotlib.pyplot as plt
arr=np.array([[1,4,8],[7,6,1],[4,5,6]])
mat=np.asmatrix(arr)
print("array",arr)
plt.matshow(mat)
plt.show()


# In[14]:


import numpy as np
a=np.array([[1,-1,2],[3,4,-5],[2,-1,3]])
b=np.array([7,-5,12])
c=np.linalg.inv(a)
x=np.dot(c,b)
print(x)
print(np.dot(a,x))


# In[15]:


import numpy as np
import matplotlib.pyplot as plt

t=np.arange(0,10,0.01)
a=np.sin(t)
b=np.cos(t)
c=np.sinh(t)
d=np.cosh(t)

plt.figure(figsize=(15,10))
plt.tight_layout

plt.subplot(2,2,1)
plt.plot(t,a,color='y')
plt.title('sine wave')
plt.xlabel('time')
plt.ylabel('amplitude')

plt.subplot(2,2,2)
plt.plot(t,b,color='g')
plt.title('cosine wave')
plt.xlabel('time')
plt.ylabel('amplitude')

plt.subplot(2,2,3)
plt.plot(t,c,color='b')
plt.title('hyp sine wave')
plt.xlabel('time')
plt.ylabel('amplitude')

plt.subplot(2,2,4)
plt.plot(t,d,color='r')
plt.title('hyp cosine wave')
plt.xlabel('time')
plt.ylabel('amplitude')


plt.show()


# In[16]:


import random
import matplotlib.pyplot as plt
import numpy as np
N=[100,500,1000,5000,500000]
abserr=[]

for i in (N):
    p=0
    for j in range(i):
        toss=random.randint(0,1)
        if(toss==1):
            p=p+1
    pHeads=p/i
    q=np.abs(0.5-pHeads)
    abserr.append(q)
print(abserr)
plt.plot(N,abserr)
plt.show()


# In[17]:


import numpy as np

t=int(input("Enter a threshold value:"))
samples=np.random.uniform(0,10,10)
ca=0
for i in (samples):
    if(i>t):
        ca=ca+1
cb=10-ca
print("thershold is ",t)
print(samples)
print("no of values above threshold",ca)
print("no of values below threshold",cb)


# In[18]:


import numpy as np
import matplotlib.pyplot as plt

def comp(n,t,T):
    terms=(1/n)*np.cos(2*np.pi*n*t/T)
    return terms
T=40
t=np.arange(0,100,0.01)
numterm=1
realize=0

for i in np.arange(1,numterm+1):
    val=(4/np.pi)*((-1)**(i-1))*comp(2*i-1,t,T)
    realize = realize+val
plt.plot(t,realize,'b')
plt.show()


# In[ ]:





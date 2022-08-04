#!/usr/bin/env python
# coding: utf-8

# In[1]:


#normal particle in a box 
from numpy import sin,pi, linspace, arange, sqrt,cos
from pylab import plot,xlabel, ylabel, legend,show, title
#Energy Eigenvalue
def E(n):
    return (n**2*pi**2*hbar**2)/(2*m*L**2)

#wave function 
def psi(n,x):
    return(sqrt(2/L)*sin((n*pi*x)/L))
#constants 
m = 9.1094e-31            #mass of electron 
hbar = 1.054571817e-34
eV = 1.6022e-19           #1 ev to J 
x0 = 0 
N = 1000
L = 1e-10                 #length of box 
dx = L/N
xvals = arange(x0, L, dx)

#first 3 energy eigenvalues
print(E(1)/eV, "eV")
print(E(2)/eV, "eV")
print(E(3)/eV, "eV")

#plot wave functions 
plot(xvals,psi(1,xvals), label = "ground state")
plot(xvals,psi(2,xvals), label = "1st excited state")
plot(xvals,psi(3,xvals), label = "2nd excited state")
xlabel("x")
ylabel("psi(x)")
title("psi(x) vs x")
legend(loc="upper right")
show()

#plot probabilities 
probability0 = psi(1,xvals)**2
probability1 = psi(2,xvals)**2
probability2 = psi(3,xvals)**2
plot(xvals,probability0, label = "ground state")
plot(xvals,probability1, label = "1st excited state")
plot(xvals,probability2, label = "2nd excited state")
xlabel("x")
ylabel("probability")
title("probability vs x ")
legend(loc="upper right")


# In[2]:


from numpy import array,exp,sqrt

# constants 
m = 9.1094e-31            # mass of electron       
hbar = 1.054571817e-34     
eV = 1.6022e-19           # 1 eV in J  
L = 1e-10                 #length of box 
N = 1000                
dx = L/N                  # step           
xvals = arange(0,L,dx)  

#potential bump constants
a = 2/3*L
w = L/10
V0 = 1e-17 # in J
Vvals = []

#potential at given point
def V(x):
    
    if 0 <= x <= L:
        return V0*exp(-(x-a)**2/(2*w**2))
    else:
        return 0

for x in xvals:
    Vvals.append(V(x))
#plot potential inside box 
plot(xvals, Vvals)
xlabel("x")
ylabel("V(x)")
title("V(x) vs x")
show()

# Solve for wave function
def solvepsi(E):
    r = array([0, 1.0], float) # r vector
    psivals = []
    
    # Solve differential equation dr/dt = f(r,t)
    def f(r,x):
        psi = r[0]
        phi = r[1]
        fpsi = phi                              # dpsi/dx
        fphi = (2*m/hbar**2)*(V(x) - E)*psi     # dphi/dx
        return array([fpsi, fphi], float)

    for x in xvals:
        psivals.append(r[0])
        # 2nd order Runge-Kutta
        xmid = x + dx/2
        rmid = r + dx/2*f(r,x)
        r += dx*f(rmid, xmid)
        #psivals.append(r[0])
    return array(psivals)
#create array of Energy Values 
Evals = linspace(0, 1000*eV,N)
psiL = []

#find values of psiL for each Energy 
for E in Evals: 
    psiL.append(solvepsi(E)[-1])      
plot(Evals,psiL)

xlabel("E")
ylabel("psiL")
title("psil(L) vs E")


# In[3]:



def f(r,x,E):
    psi = r[0]
    phi = r[1]
    fpsi = phi
    fphi = (2*m/hbar**2)*(V(x)-E)*psi
    return array([fpsi, fphi], float)

#finds wavefunction at particular point
def Solve(E): 
    r = array([0,1],float)
    for x in xvals:
        k1 = dx*f(r,x,E)
        k2 = dx*f(r+0.5*k1, x +0.5*dx ,E)
        k3 = dx*f(r+0.5*k2, x +0.5*dx,E)
        k4 = dx*f(r+k3, x+dx, E)
        r += (k1+2*k2+2*k3+k4)/6
    return r[0]
#guesses for secant method 
guesses = array([[0.06e-16, 0.104e-16],
                [0.2e-16,0.3e-16],
                [0.53e-16,0.61e-16]],float)
accuracy = 10e-3
Elist = []
#secant method to find accepted energy values 
for i in range(3):
    E1 = guesses[i,0]
    E2 = guesses[i,1]
    delta = 1
    while delta > accuracy:
        E3 = E2 - Solve(E2)*(E2-E1)/(Solve(E2)-Solve(E1))
        delta = abs(E3-E2)
        E1 = E2
        E2 = E3
    Elist.append(E2)
    print("E",i, "= {:.3f}eV".format(E2/eV))


# E0 is the ground state, E1 is first excited state, E2 is second excited state

# In[4]:


for E in Elist: 
    #not normalized wave function 
    nonormal = solvepsi(E)
    
    #probability of nonormalized wave function 
    prob = nonormal**2
    a = 0
    b = 999
    I = 0
    for k in range(1,N):
        #trapezoid rule to find integral from 0 to L of probability 
        dk = prob[k]- prob[k-1]
        I += prob[k-1]+0.5*dk*(prob[k-1]+prob[k])
        
    #normalize wave function
    normal = nonormal/I 
    
    #plot normalized wave function
    plot(xvals,normal)
    xlabel("x")
    ylabel("psi(x)")
    title("psi(x) vs x")


# In[5]:


#Focus on ground state energy
nonormal0 = solvepsi(Elist[0])
prob0 = nonormal0**2
a = 0
b = 999
I0 = 0
for k in range(1,N):
    #trapezoid rule
    dk = prob0[k]- prob0[k-1]
    I0 += prob0[k-1]+0.5*dk*(prob0[k-1]+prob0[k])
    
#normalized wave function for E0 (ground state)
normal0 = nonormal0/I0
#plot normalized wave probability
plot(xvals,normal0**2,label = "probability of ground state ")
xlabel("x")
ylabel("probability")
title("probability vs x")
legend()


# In[6]:





# In[ ]:





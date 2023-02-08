#!/usr/bin/python
"""
Plane Wave Pair Production - pwpp scripts

References:
# [1] https://journals.aps.org/pra/abstract/10.1103/PhysRevA.96.022128
"""
from tqdm import tqdm, trange
import json
import scipy.integrate as integrate
import scipy.special as special
# numpy
import numpy as np
np.random.seed(123)
from numpy.random import default_rng
rng = default_rng()
# import functions
from scipy.special import kv, iv, erf
from scipy.integrate import quad
from numpy import log, log10, sin, cos, exp, sqrt
# interpolate
from scipy import interpolate
# physical constants
from scipy.constants import pi, c, alpha, hbar, e
mGeV = 0.5109989461/1000; #[GeV]
m = mGeV;
# root finding
from scipy.optimize import fsolve
from scipy import optimize
# plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def g(x):
    """
        g function
    """
    return ( 1 + 4.8*(1+x)*log(1+1.7*x) + 2.44*x**2 )**(-2/3)

def ppm(a0,w0,n,w):
    """
        eq (4) pair creation
    """
    return alpha * a0 * n * scR((2*a0*w0*w)/(m**2))

def scR(x):
    """
        eq (5) auxiliary functional of P±
    """
    return (0.453*kv(1/3,(4)/(3*x))**2) / (1+0.145*x**(1/4)*log(1+2.26*x)+0.330*x)

def phic(g0, a0, w0, n):
    """
        eq (7) critical phase (phic is implicit function of chic)
    """
    return sqrt( (2*pi**2*n**2/log(2))*log((2*g0*a0*w0)/(chic(g0,a0,w0,n)*m)) )

def chic(g0, a0, w0, n):
    """
        eq (8) critical chi (implicit)
    """
    def nonlin(chi):
        return chi**4 * g(chi)**2 - (72*log(2))/(pi**2*alpha**2)*((g0*w0)/(n*m))**2 * log((2*g0*a0*w0)/(m*chi))
    res = fsolve(nonlin, 1e-10)
    return res

def Omega(g0,a0,w0,n):
    """
        eq (13) radiated energy
    """
    return (sqrt(2*pi)*g0*m) * ( (2*log((2*g0*a0*w0)/(m*chic(g0,a0,w0,n))))/( 1+2*log((2*g0*a0*w0)/(m*chic(g0,a0,w0,n))) ) )**(1/2)

def chicrr(g0,a0,w0,n):
    """
        eq (14) critical chi with radiation reaction
    """
    res = chic(g0,a0,w0,n) / (1+Omega(g0,a0,w0,n)/(2*g0*m)) ;
    return res

def gphi():
    """
        eq (16) electron energy assuming "radiated power and χ as functions of phase are approximately Gaussian in form"
    """
    sigma = sigsq(g0,a0,w0,n)
    return gf + ( g0*Omega/(2*g0*m+Omega) ) * (1+erf((phi-phic)/(sqrt(2)*sigma)))

def chiphi(g0,a0,w0,n,phi):
    """
        eq (17) chi in gaussian form
    """
    res = (chic(g0,a0,w0,n))/(1+(Omega(g0,a0,w0,n))/(2*g0*m)) * exp(-(phi-phic(g0,a0,w0,n))**2/(2*sigsq(g0,a0,w0,n)));
    return res

def sigsq(g0,a0,w0,n):
    """
        eq (18) sigma squared
    """
    return (pi**2*n**2)/(log(2)) * (1+2*log((2*g0*a0*w0)/(m*chic(g0,a0,w0,n))))**(-1)

def dNgdw(g0,a0,w0,n,w):
    """
        eq (20) photon spectra
    """
    chi0 = 2*g0*a0*w0/m;
    res = (sqrt(3)*pi*alpha*Fhe(g0,a0,w0,n))/(sqrt(2*log(2))) * (a0*n)/(sqrt(g0*m)) * (chicrr(g0,a0,w0,n)/chi0)/(sqrt(1+2*log(chi0/chic(g0,a0,w0,n)))) * (exp(-(2*w)/(3*chicrr(g0,a0,w0,n)*(g0*m-w))))/(sqrt(3*chicrr(g0,a0,w0,n)*(g0*m-w)+4*w));
    return res

def Fhe(g0,a0,w0,n):
    """
        eq (21) hard photon correction
    """
    arg = np.real((sqrt(2*log(2))*phic(g0,a0,w0,n)/(2*pi*n)));
    res = 0.5*(1-erf(arg));
    return res

def wc(g0, a0, w0, n):
    """
        eq (23) critical frequency
    """
    return (g0*m) * (sqrt( (2*chicrr(g0,a0,w0,n)*m)/(a0*g0*w0) ))/(1+sqrt( (2*chicrr(g0,a0,w0,n)*m)/(a0*g0*w0) ))

def Np(g0, a0, w0, n):
    """
        eq (24) positron yield
    """
    wcc = wc(g0,a0,w0,n);
    res = (3*sqrt(pi)*ppm(a0,w0,n,wcc)*chicrr(g0,a0,w0,n)/sqrt(2)) * ((g0*m-wcc)**2/(g0*m)) * dNgdw(g0,a0,w0,n,wcc);
    return res

def gp(g0, a0, w0, n):
    """
        eq (25) positron average energy
    """
    wcc = wc(g0,a0,w0,n);
    res = (wcc/(2*m))*(1+(pi**(3/2)*alpha/(3*sqrt(2*log(2))))*(n*a0**2*w0*wcc/m**2)*g(a0*w0*wcc/m**2))**(-1);
    return res

def TErber(chi):
    """
        eq (A4) Erber approximation of pair creation rate
    """
    return 0.16/chi * kv(1/3,4/(3*chi))**2

"""
Specific to reproduce plots in [1]
"""
def chicmod(g0, a0, w0, n):
    """
        critical chi mod (fig 3)
    """
    def nonlin(chi):
        return chi**4 - (72*log(2))/(pi**2*alpha**2)*((g0*w0)/(n*m))**2*log((2*g0*a0*w0)/(m*chi));
    return fsolve(nonlin, 1e-10);

def chiphimod(g0,a0,w0,n,phi):
    """
        eq (17) chi in gaussian form mod, assume Omega~0
    """
    return (chic(g0,a0,w0,n)) * exp(-(phi)**2/(2*sigsq(g0,a0,w0,n)));

def chiphimod2(g0,a0,w0,n,phi):
    """
        eq (16) and eq (6)
    """
    sigsqq = sigsq(g0,a0,w0,n);
    Omegaa = Omega(g0,a0,w0,n);
    phicc = phic(g0, a0, w0, n);
    def nonlin(chi):
        return chi**2 * g(chi) - 3*w0/(alpha*m)*( g0*Omegaa/(2*g0*m+Omegaa)*2/sqrt(2*pi*sigsqq) * exp(-(phi-phicc)**2/(2*sigsqq)) );
    return fsolve(nonlin, 1e-8);

def dNgdwmod(g0,a0,w0,n,w):
    """
        eq (20) photon spectra modified, assume chicrr->chic
    """
    chi0 = 2*g0*a0*w0/m;
    return (sqrt(3)*pi*alpha*Fhe(g0,a0,w0,n))/(sqrt(2*log(2))) * (a0*n)/(sqrt(g0*m)) * (chicrr(g0,a0,w0,n)/chi0)/(sqrt(1+2*log(chi0/chic(g0,a0,w0,n)))) * (exp(-(2*w)/(3*chic(g0,a0,w0,n)*(g0*m-w))))/(sqrt(3*chic(g0,a0,w0,n)*(g0*m-w)+4*w));

def Npmod(g0, a0, w0, n):
    """
        eq (24) positron yield without radiation reaction
    """
    wcc = wcmod(g0,a0,w0,n);
    return (3*sqrt(pi)*ppm(a0,w0,n,wcc)*chic(g0,a0,w0,n)/sqrt(2)) * ((g0*m-wcc)**2/(g0*m)) * dNgdw(g0,a0,w0,n,wcc);

def phicmod(g0, a0, w0, n):
    """
        critical phase mod (fig 3)
    """
    return sqrt( (2*pi**2*n**2/log(2))*log((2*g0*a0*w0)/(chicmod(g0,a0,w0,n)*m)) );

def wcmod(g0, a0, w0, n):
    """
        eq (23) critical frequenc, no rrr
    """
    return (g0*m) * (sqrt( (2*chic(g0,a0,w0,n)*m)/(a0*g0*w0) ))/(1+sqrt( (2*chic(g0,a0,w0,n)*m)/(a0*g0*w0) ));

def arraycenter(x):
    """
        returns centered array
    """
    return np.array([(x[i]+x[i+1])/2 for i in range(len(x)-1)])

def gauss3D(z,x,y,a0,W0,lbd):
    """
        Gaussian laser vector potential 3D
    """
    zR = pi*W0**2/lbd
    return a0/sqrt(1+(z/zR)**2) * exp(-(x**2+y**2)/(W0**2*(1+(z/zR)**2)))

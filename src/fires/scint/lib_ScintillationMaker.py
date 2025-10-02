# Adapted for FIRES from the ScintillationMaker project:
#   https://github.com/SprengerT/ScintillationMaker/tree/main
# Original C++/Python implementation by SprengerT (see upstream repository for full details).

import numpy as np
import os

from numpy.ctypeslib import ndpointer
import ctypes

#import C++ library
dir_path = os.path.dirname(os.path.realpath(__file__))
file_c = os.path.join(dir_path,"ScintillationMaker.so")
lib = ctypes.CDLL(file_c)

#load C++ library for fast SP trafo
lib.simulate_SimpleScreen.argtypes = [
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_real [N_t*N_nu]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_im [N_t*N_nu]
    ctypes.c_int,   # N_t
    ctypes.c_int,   # N_nu
    ctypes.c_int,   # N_im
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # nu_scaled [N_nu]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # t_scaled [N_t]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # thx [N_th]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # thy [N_th]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # mu [N_th]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # ph [N_th]
]

def simulate_scintillation(t,nu,t_s,nu_s,N_im=1000,th_lim=3., ref_freq_hz=1000e6):
    
    thx = np.random.uniform(-th_lim,th_lim,N_im)
    thy = np.random.uniform(-th_lim,th_lim,N_im)
    mu = np.exp(-0.5*(thx**2+thy**2))
    ph = 2.*np.pi*np.random.random(N_im)
    
    N_t = len(t)
    N_nu = len(nu)
    
    t_scaled = -nu_s*t/(2.*ref_freq_hz*t_s)
    nu_scaled = nu/nu_s
    
    E_real = np.zeros(N_t*N_nu,dtype=float)
    E_im = np.zeros(N_t*N_nu,dtype=float)
    lib.simulate_SimpleScreen(E_real,E_im,N_t,N_nu,N_im,nu_scaled,t_scaled,thx,thy,mu,ph)
    E = E_real.reshape((N_t,N_nu))+1.0j*E_im.reshape((N_t,N_nu))
    
    return E
import numpy as np
from .mcmc import mcmc
from .orderfit import parcalc
import matplotlib.pyplot as plt
from astropy.io import fits
import os


def fit(phase, rv, sigrv, ldcoef,lderr,data_type,tag):
    from .constants import storbitpar, Wguess, spect, Wpar, Wpriors,Wsingle
    if storbitpar["ld_law"]=="ln" and storbitpar["model"]=="pyastronomy":
        Wguess["ldc"]=ldcoef[0]
        Wpriors["ldc"]=[ldcoef[0],lderr[0]]

    elif storbitpar["ld_law"]=="ln" and storbitpar["model"]=="pyarome":
        Wguess["ldc1"]=ldcoef[0]
        Wpriors["ldc"]=[ldcoef[0],lderr[0]]

    elif storbitpar["ld_law"]=="qd":
        Wguess["ldc1"],Wguess["ldc2"]=ldcoef
        Wpriors["ldc1"],Wpriors["ldc2"]=[[ldcoef[k],lderr[k]] for k in range(len(ldcoef))] 

    elif storbitpar["ld_law"]=="nl":
        Wpriors["ldc1"],Wpriors["ldc2"],Wpriors["ldc3"],Wpriors["ldc4"]=[[ldcoef[k],lderr[k]] for k in range(len(ldcoef))]
        Wguess["ldc1"],Wguess["ldc2"],Wguess["ldc3"],Wguess["ldc4"]=ldcoef

    else:
        sys.exit("Non-existent limb-darkening law for that model!")
    
    pout = mcmc(rv, phase, sigrv,Wguess,Wpar,Wpriors,Wsingle, data_type,tag)
    return(pout)

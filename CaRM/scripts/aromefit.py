import numpy as np
import pyximport; pyximport.install()
from .Pyarome import arome
from PyAstronomy import modelSuite as ms
from .constants import storbitpar
import matplotlib.pyplot as plt


def kepler(x, vsys, k):
    n = len(x)
    return(np.array(vsys*np.ones(n))+np.array([k*np.sin(2.*np.pi*j+np.pi) for j in x]))


def fitmodel(x, pari):
    #print("-----------------")
    #print(pari)
    pari=pari.values()
    x = np.array(x)
    model = storbitpar["model"]
    P = float(storbitpar["P"])
    Rstar = float(storbitpar["Rstar"])
    tepoch = float(storbitpar["tepoch"])
    if model == "pyarome":
        #print(pari)
        #print(len(pari))
        #print("-----------------")
        [vsys, rp, k, sma, inc, lda, ldc1, ldc2,ldc3,ldc4,beta0,Vsini, sigma0, zeta, Kmax, dT0,sigw,aslope,ln_a,ln_tau] = pari
        x = x+np.ones(len(x))*dT0
        return(kepler(x, vsys, k)+np.array([arome(y*360.+90., sma, inc, lda, np.array([ldc1, ldc2]), beta0, Vsini, sigma0, zeta, rp, Kmax, units='degree') for y in x]),aslope,sigw,ln_a,ln_tau)

    elif model == "pyastronomy":
        [vsys, rp, k, sma, inc, lda, ldc1, Vrot, Is, Omega, dT0,sigw,aslope,ln_a,ln_tau] = pari
        # phase=(x+np.ones(len(x))*dT0)*P+tepoch*np.ones(len(x))
        # print(phase)
        # sys.exit()
        x = x+np.ones(len(x))*dT0
        rmcl = ms.RmcL()
        rmcl.assignValue({"a": float(sma), "lambda": float(lda/180.0*np.pi), "epsilon": float(ldc1),
                          "P": float(P), "T0": float(tepoch), "i": float(inc/180.*np.pi),
                          "Is": float(Is/180.0*np.pi), "Omega": Omega, "gamma": float(rp)})
        # x=np.array(x)
        rv = rmcl.evaluate(tepoch*np.ones(len(x))+P*x)
        #rv = rmcl.evaluate(phase)
        return(kepler(x, vsys, k)+np.array(rv*Rstar/1000.),aslope,sigw,ln_a,ln_tau)

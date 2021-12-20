import numpy as np


def Uprior(x, interval):
    # delta=abs(max(interval)-min(interval))
    if x > min(interval) and x < max(interval):
        return(0.)
    else:
        return(-np.inf)


def Gprior(x, gpar):
    Gmu, Gsig = gpar
    return((-(x-Gmu)**2.)/(2.*Gsig**2.)-0.5*np.log(2.*np.pi)-2.*np.log(Gsig))

from .priors import*
from .aromefit import*
import numpy as np
# from .constants import parguess#,cpar,prior_type,prior_int
from copy import deepcopy as dpcy
import matplotlib.pyplot as plt
import time
import sys
import scripts.globalvar as gb
import os
import time

def lnlike(par, t, rv, yerr):
    model,_,sigw,_,_=fitmodel(t, par)
    inv_sigma2 = 1.0/(np.array(yerr)**2. + np.exp(2.*sigw))
    return -0.5*(np.sum((rv-model)**2.*inv_sigma2 - np.log(inv_sigma2)))


def lnprior(par, prior_type, prior_interval):
    pfun=0
    if str(prior_type)=="U":
        pfun+=Uprior(par,prior_interval)
    elif str(prior_type)=="G":
        pfun+=Gprior(par,prior_interval)
    else:
        None
    return(pfun)


import scripts.globalvar as gb
def lnprob(par):
    
    rv, ph, sigrv,guessdict,pardict,priordict,odict,dlen,outoftransitph,outoftransitrv,outoftransitsigrv=dpcy(gb.interpar)

    tempdict=dict()
    c=0
    for key in odict:    
        tempdict[key]=par[c]
        c+=1

    fprob=0
    for k in range(dlen):
        lnprior_val=0
        intdict=dpcy(guessdict[k])
        for j in odict:
            try:
                lnprior_val+= lnprior(tempdict[j],pardict[k][str(j)],priordict[k][str(j)])
                intdict[str(j)]=dpcy(tempdict[j])
            except:
                None
        
        if np.isfinite(lnprior_val)==False:
            return(-np.inf)
        else:
            None
        ll=lnlike(intdict, ph[k], rv[k], sigrv[k])   

        fprob+=lnprior_val+ll

        if np.isnan(fprob) == 1:
            return(-np.inf)

    return(fprob)

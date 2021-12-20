import numpy as np
import random
from .aromefit import fitmodel
#from .probfun_md import lnprior, lnlike
import scipy.optimize as op
import emcee
import sys
import matplotlib.pyplot as plt
from copy import deepcopy as dpcy
import os
from astropy.io import fits
import string
import json
from multiprocessing import Pool
from scripts.lnpost_selection import lnposterior_selection
# from emcee_tools import
from scripts.constants import gps, Wguess

doslope=Wguess["act_slope"]

optkey=0

if gps==True:
    if doslope!=None:
        from .probfun_GPs_slope import lnprior, lnlike, lnprob
        optkey=0
    elif doslope==None:
        from .probfun_GPs import lnprior, lnlike, lnprob
        optkey=1
else:
    if doslope!=None:
        from .probfun_slope import lnprior, lnlike, lnprob
        optkey=2
    elif doslope==None:
        from .probfun import lnprior, lnlike, lnprob
        optkey=3
    


def chain_corner(sampler):
    qntls = [0.15865, 0.5, 0.84135]
    isig_fact = 4.
    iquantile = 90.
    iquantile_walker = 1.
    iverbose = 1
    varval = "median" 
  
    lnprob=sampler.lnprobability
    indexes = lnposterior_selection(lnprob, isig_fact, iquantile, iquantile_walker, iverbose)
    samplerchain = sampler.chain[indexes[0]]
    flat = []
    for chain in samplerchain:
        for pars in chain:
            flat += [pars]

    flatchain = dpcy(flat)
    lnprob=lnprob[indexes[0]]

    real = [(v[1], v[2]-v[1], v[1]-v[0]) for v in zip(*np.quantile(flatchain, qntls, axis=0))]
    if varval=="max":
        maxprob = lnprob.ravel().max()
        parmax = dpcy(flatchain[np.nonzero(lnprob.ravel() == maxprob)[0][0]])
    elif varval=="median":
        parmax = dpcy([real[x][0] for x in range(len(real))])
    return(parmax)

def mcmc(data,dataph,sig_err,guess,par,priors,single,data_type,tag):
    print("#####")
    print(guess)
    dlen = len(data)
    
    if optkey==0 or optkey==2:
        from scripts.constants import storbitpar, useall
        dTransit= storbitpar["delta_transit"]/(storbitpar["P"]*2)
        outoftransitph=[]
        outoftransitrv=[]
        outoftransitsigrv=[]
        if useall==True:
            outoftransitph=dpcy(dataph)
            outoftransitrv=dpcy(data)
            outoftransitsigrv=dpcy(sig_err)
        else:
            for k in range(dlen):
                outtphi=dpcy(dataph[k])
                outtrvi=dpcy(data[k])
                outtsigrvi=dpcy(sig_err[k])
                a=[j>dTransit for j in outtphi]
                b=[j<(-dTransit) for j in outtphi]
                indexes=[j or i for (j,i) in zip(a,b)]
                outoftransitph+=[outtphi[indexes]]
                outoftransitrv+=[outtrvi[indexes]]
                outoftransitsigrv+=[outtsigrvi[indexes]]
    
    else:
        outoftransitph,outoftransitrv,outoftransitsigrv=[None,None,None]
    """
    for k in range(dlen):
        plt.plot(outoftransitph[k],outoftransitrv[k],"*r")
        plt.plot(dataph[k],data[k],".b")
    plt.show()
    """
    from .constants import ncores
    # Initialization: Reads the constants file to get chains parameters and variables
    if data_type == 0:
        from .constants import Wnprod, Wnburn, Wnchains,vardict_plot
        nprod, nburn, nchains = Wnprod, Wnburn, Wnchains

        guessdict=[dict() for x in range(dlen) ]
        pardict=[dict() for x in range(dlen) ]
        priordict=[dict() for x in range(dlen) ]
        var_plot=dict()
        vardict=[[] for x in range(dlen) ]
        

        for i in range(dlen):
            guessi=guessdict[i]
            pari=pardict[i]
            priori=priordict[i]
            vdict=vardict[i]
            for key in priors:
                if single[key]==True:
                     guessi[key+"_"+str(i)]=guess[key]
                     pari[key+"_"+str(i)]=par[key]
                     priori[key+"_"+str(i)]=priors[key]
                     vdict+=[key+"_"+str(i)]
                     var_plot[key+"_"+str(i)]=vardict_plot[key]+r"$_{\,\,"+str(i)+"}$"
                     
                else:
                     guessi[key]=guess[key]
                     pari[key]=par[key]
                     priori[key]=priors[key]
                     vdict+=[key]
                     var_plot[key]=vardict_plot[key]
    else:
        from .constants import Cnprod, Cnburn, Cnchains,vardict_plot
        nprod, nburn, nchains = Cnprod, Cnburn, Cnchains

        guessdict=dpcy(guess)
        pardict=[dict() for x in range(dlen) ]
        priordict=[dict() for x in range(dlen) ]

        var_plot=dict()
        vardict=[[] for x in range(dlen) ]
        

        for i in range(dlen):
            #guessi=guessdict[i]
            pari=pardict[i]
            priori=priordict[i]
            vdict=vardict[i]
            for key in priors:
                if single[key]==True:
                     #guessi[key+"_"+str(i)]=guess[key]
                     pari[key+"_"+str(i)]=par[key]
                     priori[key+"_"+str(i)]=priors[key]
                     vdict+=[key+"_"+str(i)]
                     var_plot[key+"_"+str(i)]=vardict_plot[key]+r"$_{\,\,"+str(i)+"}$"
                else:
                     #guessi[key]=guess[key]
                     pari[key]=par[key]
                     priori[key]=priors[key]
                     vdict+=[key]
                     var_plot[key]=vardict_plot[key]


    from .random_chain import random_chain as rd
    p0,odict=rd(par, priors,guess,single,nchains,dlen)


    import scripts.globalvar as gb
    gb.interpar=[data, dataph,sig_err,guessdict,pardict,priordict,odict,dlen,outoftransitph,outoftransitrv,outoftransitsigrv]


    ndim=int(len(odict))
    with Pool(ncores) as pool:
        sampler = emcee.EnsembleSampler(int(nchains), ndim, lnprob, pool=pool)
        print("Running burn-in...")
        p1,lp,_= sampler.run_mcmc(p0, nburn, progress=True)

        maxprob=p1[np.argmax(lp)]
        maxstd=[np.std(k) for k in p1.T]
        for k in range(len(p1.T)):
            p1.T[k]= np.random.normal(maxprob[k],maxstd[k]/3.,int(nchains))

        sampler.reset()
        p1= sampler.run_mcmc(p1, int(nburn), progress=True)
        burn_sampler=sampler

    with Pool(ncores) as pool:
        sampler = emcee.EnsembleSampler(int(nchains), ndim, lnprob, pool=pool)
        print("Running production")
        p0= sampler.run_mcmc(p1, int(nprod), progress=True)

    emcee_trace = sampler.chain[:, :, :].reshape(-1, ndim).T

    #real = [(v[1], v[2]-v[1], v[1]-v[0]) for v in zip(*np.percentile(sampler.flatchain, [15.865, 50, 84.135], axis=0))]



    from scripts.constants import parmaxprob
    
    if parmaxprob==True:
        # Compute the maximum probability parameters
        maxprob = sampler.lnprobability.ravel().max()
        parmax = sampler.flatchain[:, :][np.nonzero(
            sampler.lnprobability.ravel() == maxprob)[0][0]]
    else:
        parmax=chain_corner(sampler)

    # Update the guesses
    c=0
    tempdict=dict()
    for key in odict:    
        tempdict[key]=parmax[c]
        c+=1

    newdict=[]
    for k in range(dlen):
        intdict=dpcy(guessdict[k])
        for j in odict:
            try:
                pardict[k][str(j)]
                guessdict[k][j]=tempdict[j]
            except:
                None

    import pickle
    savefile=open( "data.pkl", "rb" )
    saved_dict=pickle.load(savefile)
    savefile.close()
    savefile=open( "data.pkl", "wb" )
    savedata=dict()
    savedata["FITTED_PAR"]=odict
    savedata["PAR_DICT"]=pardict
    savedata["PRIOR_DICT"]=priordict
    savedata["PRIOR_NAM"]=var_plot
    savedata["FITTED_PAR_EST"]=tempdict
    savedata["HIGHEST_LNLIKE_PAR"]=guessdict
    savedata["BURN_SAMPLER"]=burn_sampler
    savedata["SAMPLER"]=sampler
    savedata["DATA"]=[data,dataph,sig_err]
    saved_dict["RUNS"][tag]=savedata

    pickle.dump(saved_dict,savefile)
    savefile.close()
    print(guessdict)
    return(guessdict)


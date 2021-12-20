#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import emcee
import matplotlib.pyplot as plt
from scripts.aromefit import fitmodel, kepler
from scripts.lnpost_selection import lnposterior_selection
from scripts.corner import*
from scripts.plot_chains import*
from scripts.aromefit import fitmodel
from copy import deepcopy as dpcy
plt.ticklabel_format(useOffset=False)
import celerite
from celerite import terms
import ast
import arviz
import pandas
import random
sdict=pickle.load(open("data.pkl","rb"))


#######User imput#######
# Quantiles to be used for the error bars
qntls = [0.15865, 0.5, 0.84135]
isig_fact = 5.
iquantile = 95.
iquantile_walker = 0.5
iverbose = 0
varval = "median"
docorner = 1
dochains=1
nsamples=300
gps=False
#######################





def get_priors(tempdict,pardict,prior_dist,bound):
    # Prior distributions
    fkeys=tempdict.keys()
    priordict={}
    vout={}
    priorintdict={}
    pdist={}
    for key in fkeys:
        vout[key]=varnames[key]
        for night in range(nights):
            try:
                pardict[night][key]
                priordict[key]=pardict[night][key]
                pdist[key]=prior_dist[night][key]
                priointdict[key]=pardict[night][key]

            except:
                None

    npoints=5000

    priorchains=[]
    k=0
    for key in priordict.keys():
        if priordict[key]=="U":
            if pdist[key][0]< bound[k][0]:
                l=bound[k][0]
                if pdist[key][1]> bound[k][1]:
                    h=bound[k][1]
                else:
                    h=pdist[key][1]
            else:
                l=pdist[key][0]
                if pdist[key][1]> bound[k][1]:
                    h=bound[k][1]
                else:
                    h=pdist[key][1]
            priorchains+=[np.random.uniform(low=l,high=h, size=npoints)]
        elif priordict[key]=="G":
            priorchains+=[np.random.normal(pdist[key][0],pdist[key][1],size=npoints)]
        k+=1
    
    
    return(np.array(priorchains).T,vout)


# In[3]:


def upfitlin(phf,par,phi,rvi,sigrvi):
    # Fitted model
    f,slope,sigw,ln_a,ln_tau=fitmodel(phf, par)
    if slope==None:
        slope=0
    else: 
        None
    return(f+slope*phf)


if gps==True:
    def upfit(phf,par,phi,rvi,sigrvi):
        # Fitted model
        f,slope,sigw,ln_a,ln_tau=fitmodel(phf, par)
        if slope==None:
            slope=0
        else: 
            None
        kernel = terms.RealTerm(ln_a,ln_tau)
        gp=celerite.GP(kernel)

        gp.compute(phi, sigrvi)

        m = gp.predict(rvi -fitmodel(phi,par)[0]-slope*phi, phf )[0] + f+ slope*phf
        return(m)


    def upfit2(phf,par,phi,rvi,sigrvi):
        # Fitted model
        f,slope,sigw,ln_a,ln_tau=fitmodel(phf, par)
        if slope==None:
            slope=0
        else: 
            None
        kernel = terms.RealTerm(ln_a,ln_tau)
        gp=celerite.GP(kernel)

        gp.compute(phi, sigrvi)

        m = gp.predict(rvi -fitmodel(phi,par)[0]-slope*phi, phf )[0]
        return(m)

    def upfit3(phf,par,phi,rvi,sigrvi):
        # Fitted model
        m,slope,sigw,ln_a,ln_tau=fitmodel(phf, par)
        if slope==None:
            slope=0
        else: 
            None

        return(m+slope*phf)
else:
    def upfit(phf,par,phi,rvi,sigrvi):
        # Fitted model
        f,slope,sigw,ln_a,ln_tau=fitmodel(phf, par)
        if slope==None:
            slope=0
        else: 
            None
        #print(slope)
        return(f+slope*phf)


    def upfit2(phf,par,phi,rvi,sigrvi):
        # Fitted model
        f,slope,sigw,ln_a,ln_tau=fitmodel(phf, par)
        if slope==None:
            slope=0
        else: 
            None
        return(f+slope*phf)

    def upfit3(phf,par,phi,rvi,sigrvi):
        # Fitted model
        f,slope,sigw,ln_a,ln_tau=fitmodel(phf, par)
        if slope==None:
            slope=0
        else: 
            None
        return(f+slope*phf)


# In[4]:


def par_replace(guessdict,fitted_pars,pardict,parmax,nights):
    pc=dpcy(parmax)
    # Update the guesses
    c=0
    tempdict=dict()
    for key in fitted_pars:    
        tempdict[key]=pc[c]
        c+=1

    newdict=[]
    for k in range(nights):
        for j in fitted_pars:
            try:
                pardict[k][str(j)]
                guessdict[k][j]=tempdict[j]
            except:
                None
    return(guessdict)


# In[5]:


def chain_corner(sampler,burn_sampler,tag,label,tempdict,pardict,prior_dist):

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
        for x in range(len(real)):
            p,_,_ =real[x]
            p=parmax[x]
    elif varval=="median":
        parmax = dpcy([real[x][0] for x in range(len(real))])


    ft=np.array(flatchain).T
    intervals=[[np.median(ft[k])-6.*np.std(ft[k]), np.median(ft[k])+6.*np.std(ft[k])] for k in range(len(np.array(flatchain).T))]
    priorchains, varnames=get_priors(tempdict,pardict,prior_dist,intervals)
    vnames=list(varnames.values())
    ######Final updated variables######
    #varup = np.multiply(parm, nsubs_par) + \
    #    np.multiply(dlen*[x], (dlen*[subs_par]))
    #parf += [varup]

    g = open("out_varval.txt", "a+")
    for x in range(len(real)):
        g.write(vnames[x]+"&"+str(real[x][0])+"&" +
                str(real[x][1])+"&"+str(real[x][2])+"\n")
    g.write("\n")
    g.close()

    if docorner == 1:
        #fig=corner.corner(priorchains, color=(0.31, 0.45, 0.62, 0.5),show_titles=False,plot_countours=False,weights=np.ones(len(priorchains))/len(priorchains),range=intervals, bins=30, smooth=1.)
        fig=corner(flatchain, truths=parmax, quantiles=qntls, show_titles=True, color="k",title_fmt=".4f", title_kwargs={"fontsize": 14},plot_countours=True, labels=vnames,weights=np.ones(len(flatchain))/len(flatchain), range=intervals, bins=30)

        plt.savefig("Corner plot for the bin " + tag +".pdf", transparent=True,bbox_inches='tight')
        plt.close()
    
    if dochains==1:
        ######Plot production chains######
        plot_chains(samplerchain,lnprob, truths=parmax,l_param_name=vnames)
        plt.ticklabel_format(axis='both', style='sci', scilimits=(-2, 3))
        plt.savefig("Production for the bin " + tag +".pdf", transparent=True,bbox_inches='tight')
        plt.close()
    
        ######Plot burn-in chains######
        plot_chains(burn_sampler.chain,burn_sampler.lnprobability,l_param_name=vnames)
        plt.ticklabel_format(axis='both', style='sci', scilimits=(-2, 3))
        plt.savefig("Burn-in for the bin" + tag +".pdf", transparent=True,bbox_inches='tight')
        plt.close()
    else:
        None
    #print(np.array(flatchain).T)
    # Sample from the 1sigma samples
    #sorted_lnprob=sorted(zip(lnprob.ravel(),np.arange(0,len(lnprob.ravel()),1)))
    lnprob_mean,lnprob_std=[np.mean(lnprob.ravel()),np.std(lnprob.ravel())]
    sigma1_indexes=np.where((lnprob.ravel()>(lnprob_mean-lnprob_std))&(lnprob.ravel()<(lnprob_mean+lnprob_std)))
    #print(lnprob.ravel()[sigma1_indexes])
    clean_flatchain=dpcy(flatchain)
    rind=np.random.choice(sigma1_indexes[0],nsamples)
    return(parmax,[clean_flatchain[k] for k in rind],real)



rpsave=[]

nights=sdict["INFO"]["NUMBER_OBS"]

parestsave=[]


# In[ ]:


for wbin in sdict["INFO"]["BINS"]:

    ########## READ THE DATA FROM save.pkl ##########
    obs_data=sdict["RUNS"][wbin]["DATA"]
    sampler=sdict["RUNS"][wbin]["SAMPLER"]
    burn_sampler=sdict["RUNS"][wbin]["BURN_SAMPLER"]
    fitted_pars=sdict["RUNS"][wbin]["FITTED_PAR"]
    tempdict= sdict["RUNS"][wbin]["FITTED_PAR_EST"]
    guessdict=sdict["RUNS"][wbin]["HIGHEST_LNLIKE_PAR"]
    pardict=sdict["RUNS"][wbin]["PAR_DICT"]
    varnames=sdict["RUNS"][wbin]["PRIOR_NAM"]
    prior_dist=sdict["RUNS"][wbin]["PRIOR_DICT"]
    #################################################


    parmax,parsample,pi=dpcy(chain_corner(sampler,burn_sampler,wbin,fitted_pars,tempdict,pardict,prior_dist))
    parestsave+=[pi]

    nguessdict=dpcy(par_replace(guessdict,fitted_pars,pardict,parmax,nights))

    sampledir=[]
    for k in range(nsamples):
        d=dpcy(par_replace(guessdict,fitted_pars,pardict,parsample[k],nights))
        sampledir+=[d]
    
    #plt.style.use(['science'])
    rv,phase,sigrv=obs_data

    joint_phase=[]
    joint_rv=[]
    joint_rv_res=[]
    joint_rv_res2=[]
    joint_sigrv=[]
    joint_unsigrv=[]
    for night in range(nights):
        rvi,phasei,sigrvi=[rv[night],phase[night],sigrv[night]]
        phasen=np.linspace(min(phasei)*1.05,max(phasei)*1.05,600)
        fig = plt.figure(figsize=(5,3))
        ax1=fig.add_subplot()
        def mod_sample(k):
            return(upfit(phasen,sampledir[k][night],phasei,rvi,sigrvi))


        plt.style.use('seaborn-paper')
        # Fitted model
        fmodel,_,sigw,_,_=dpcy(fitmodel(phasen,nguessdict[night]))

        # Computes de transit duration
        tr_dict=nguessdict[night]
        tr_dur=1./np.pi * np.arcsin(1./tr_dict["sma"] *np.sqrt((1+tr_dict["rp"] )**2.-tr_dict["sma"]**2. *np.cos(np.radians(tr_dict["inc"]))**2.))
        tr_ingress_egress=1./np.pi * np.arcsin(1./tr_dict["sma"] *np.sqrt((1.-tr_dict["rp"])**2.-tr_dict["sma"]**2. *np.cos(np.radians(tr_dict["inc"]))**2.))
        m=upfit(phasen,nguessdict[night],phasei,rvi,sigrvi)
        # Plots the transit ingress and egress
        ax1.axvspan(-tr_dur/2.-tr_dict["dT0"], tr_dur/2.-tr_dict["dT0"], alpha=0.3, color='orange')
        ax1.axvspan(-tr_ingress_egress/2.-tr_dict["dT0"], tr_ingress_egress/2.-tr_dict["dT0"], alpha=0.4, color='orange')

        import multiprocessing as mp
        N= mp.cpu_count()
        with mp.Pool(processes = N) as p:
            results = p.map(mod_sample, range(nsamples))
        for k in range(nsamples):
            ax1.plot(phasen,results[k], color="gray", alpha=0.1,zorder=5)

        ax1.errorbar(phasei,rvi,sigrvi,fmt="b.", ecolor="green", capsize=2.,ms=2,capthick=0.5,elinewidth=0.7,zorder=10)
        ax1.plot(phasen,m,color="r",linewidth=0.5,zorder=6)

        uncorr_err=sigrvi
        corr_err=np.sqrt(sigrvi**2.+np.exp(sigw)**2.)

        ax1.errorbar(phasei,rvi,corr_err,fmt="k.", ecolor="k", capsize=1.,ms=0,capthick=0.5,elinewidth=0.7,zorder=9)
        ax1.set_ylabel("RV [Km/s]")
        ax1.set_xlabel("Phase")

        ax1.set_xlim([min(phasei)*1.05,max(phasei)*1.05])
        plt.savefig("Individual_fit_"+wbin+"night_"+str(night)+".pdf", transparent=True,bbox_inches='tight')

        plt.close()

        fig = plt.figure(figsize=(5,3))
        ax1=fig.add_subplot()
        kguess=dpcy(nguessdict[night])

        rpi=""
        for key in kguess:
            if key.startswith("rp")==True:  
                rpi=key
        rpsave+=[dpcy(kguess[rpi])]
        kguess[rpi]=1e-6
        kf,kslope,_,_,_=fitmodel(phasei,kguess)
        if kslope==None:
            kslope=0.
        else:
            None 
        fittedmodel=fitmodel(phasei,kguess)[0]+phasei*kslope
        rvi_kp=1000.*(rvi-fittedmodel)

        ax1.errorbar(phasei,rvi_kp,1000.*corr_err,fmt="b.", ecolor="k", capsize=1.,ms=0,capthick=0.5,elinewidth=0.7,zorder=9)
        ax1.errorbar(phasei,rvi_kp,1000.*uncorr_err,fmt="k.",  ecolor="green", capsize=2.,ms=2,capthick=0.5,elinewidth=0.7,zorder=10)
        ###
        rvfitk=upfitlin(phasen,kguess,_,_,_)

        def mod_sample(k):
            return(1000*(upfit(phasen,sampledir[k][night],phasei,rvi,sigrvi)-rvfitk))

        import multiprocessing as mp
        N= mp.cpu_count()
        with mp.Pool(processes = N) as p:
            results = p.map(mod_sample, range(nsamples))
        for k in range(nsamples):
            ax1.plot(phasen,results[k], color="gray", alpha=0.1)
        
        
        #for k in range(nsamples):
        #    try: 
        #        ax1.plot(phasen,1000*(upfit(phasen,sampledir[k][night],phasei,rvi,sigrvi)-rvfitk), color="gray", alpha=0.1)
        #    except:
        #        None
        ax1.plot(phasen,1000*(m-rvfitk),color="r",linewidth=0.5)
        plt.style.use('seaborn-paper')
        ax1.set_ylabel("RV [m/s]")
        ax1.set_xlabel("Phase")
        # Plots the transit ingress and egress
        ax1.axvspan(-tr_dur/2.-tr_dict["dT0"], tr_dur/2.-tr_dict["dT0"], alpha=0.3, color='orange')
        ax1.axvspan(-tr_ingress_egress/2.-tr_dict["dT0"], tr_ingress_egress/2.-tr_dict["dT0"], alpha=0.4, color='orange')
        ax1.set_xlim([min(phasei)*1.05,max(phasei)*1.05])
        plt.savefig("Individual_fit_"+wbin+"night_"+str(night)+"Keplerian_corrected.pdf", transparent=True,bbox_inches='tight')
        plt.close()
        

        rvi_err=1000.*(rvi-upfit3(phasei,nguessdict[night],phasei,rvi,sigrvi))
        rvi_err2=1000.*(rvi-upfit(phasei,nguessdict[night],phasei,rvi,sigrvi))

        ######Joint residuals######
        std_finalerror =   round(np.mean(1000*corr_err),1)
        std_residuals = round(np.std(rvi_err), 1)

        fig = plt.figure(figsize=(5,3))
        ax1=fig.add_subplot()
        plt.style.use('seaborn-paper')

        ax1.set_title(r"$\sigma_{res}$"+" = "+str(std_residuals) +
                        "; "r"$\sigma_W$"+" = "+str(round(1000*np.exp(sigw),1))+
                      "; "r"$\sigma$"+" = "+str(std_finalerror))
        #ax1.axhline(0.,linestyle="dotted",color="darkgreen", alpha=0.5)
        #ax1.axhline(std_finalerror,linestyle="dotted",color="green", alpha=0.33)
        #ax1.axhline(-std_finalerror,linestyle="dotted",color="green", alpha=0.33)
        #ax1.axhline(2.*std_finalerror,linestyle="dotted",color="green", alpha=0.66)
        #ax1.axhline(-2.*std_finalerror,linestyle="dotted",color="green", alpha=0.66)
        #ax1.axhline(3.*std_finalerror,linestyle="dotted",color="green", alpha=1)
        #ax1.axhline(-3.*std_finalerror,linestyle="dotted",color="green", alpha=1)



        ax1.errorbar(phasei,rvi_err,1000.*corr_err,fmt="k.", ecolor="k", capsize=1.,ms=0,capthick=0.5,elinewidth=0.7,zorder=9)
        ax1.errorbar(phasei,rvi_err,1000.*uncorr_err,fmt="b.", ecolor="green", capsize=2.,ms=2,capthick=0.5,elinewidth=0.7,zorder=10)
        if gps==True:
            ax1.plot(phasen,1000.*upfit2(phasen,nguessdict[night],phasei,rvi,sigrvi),color="r",linewidth=0.5)
        else:
            None
        ax1.set_ylabel("RV [m/s]")
        ax1.set_xlabel("Phase")
        ax1.set_xlim([min(phasei)*1.05,max(phasei)*1.05])
        # Plots the transit ingress and egress
        ax1.axvspan(-tr_dur/2.-tr_dict["dT0"], tr_dur/2.-tr_dict["dT0"], alpha=0.3, color='orange')
        ax1.axvspan(-tr_ingress_egress/2.-tr_dict["dT0"], tr_ingress_egress/2.-tr_dict["dT0"], alpha=0.4, color='orange')
        """
        ax2=fig.add_subplot(122)
        ax2.hist(rvi_err,bins=np.array([-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2.,2.5,3.])*std_finalerror, histtype='step',align="left",color="k",density=True)
        ax2.axvline(0,linestyle="dotted",color="k", alpha=1)
        """
        plt.savefig("Individual_fit_"+wbin+"night_"+str(night)+"_residuals.pdf", transparent=True,bbox_inches='tight')
        plt.close()
        #########SAVE JOINT DATA######### 
        joint_rv+=np.ndarray.tolist(rvi_kp)
        joint_phase+=np.ndarray.tolist(phasei)
        joint_rv_res+=np.ndarray.tolist(rvi_err)
        joint_rv_res2+=np.ndarray.tolist(rvi_err2)
        joint_sigrv+=np.ndarray.tolist(1000*uncorr_err)
        joint_unsigrv+=np.ndarray.tolist(1000*corr_err)

    ##############################################################

    ######Joint residuals######
    std_finalerror = round(np.mean(joint_unsigrv), 1)
    std_residuals = round(np.std(joint_rv_res), 1)

    fig = plt.figure(figsize=(5,3))
    ax1=fig.add_subplot()

    ax1.set_title(r"$\sigma_{res}$"+" = "+str(std_residuals) +
                  "; "r"$\sigma$"+" = "+str(std_finalerror))
    #ax1.axhline(0.,linestyle="dotted",color="green", alpha=0.6)
    #ax1.axhline(std_finalerror,linestyle="dotted",color="green", alpha=0.33)
    #ax1.axhline(-std_finalerror,linestyle="dotted",color="green", alpha=0.33)
    #ax1.axhline(2.*std_finalerror,linestyle="dotted",color="green", alpha=0.66)
    #ax1.axhline(-2.*std_finalerror,linestyle="dotted",color="green", alpha=0.66)
    #ax1.axhline(3.*std_finalerror,linestyle="dotted",color="green", alpha=1)
    #ax1.axhline(-3.*std_finalerror,linestyle="dotted",color="green", alpha=1)
    plt.style.use('seaborn-paper')
# Plots the transit ingress and egress
    ax1.axvspan(-tr_dur/2.-tr_dict["dT0"], tr_dur/2.-tr_dict["dT0"], alpha=0.3, color='orange')
    ax1.axvspan(-tr_ingress_egress/2.-tr_dict["dT0"], tr_ingress_egress/2.-tr_dict["dT0"], alpha=0.4, color='orange')
    ax1.errorbar(joint_phase,joint_rv_res,joint_unsigrv,fmt="k.", ecolor="k", capsize=1.,ms=0,capthick=0.5,elinewidth=0.7,zorder=9)
    ax1.errorbar(joint_phase,joint_rv_res,joint_sigrv,fmt="b.", ecolor="green", capsize=2.,ms=2,capthick=0.5,elinewidth=0.7,zorder=10)
    ax1.set_ylabel("RV [m/s]")
    ax1.set_xlabel("Phase")
    ax1.set_xlim([min(joint_phase)*1.05,max(joint_phase)*1.05])
    plt.savefig("Joint_residuals_"+wbin+".pdf", transparent=True,bbox_inches='tight')


    plt.close()
    plt.style.use('seaborn-paper')
    
    #############################################################################################################
    fig = plt.figure(figsize=(5,3))
    ax1=fig.add_subplot()
    rvi,phasei,sigrvi=[np.array(joint_rv),np.array(joint_phase),np.array(joint_unsigrv)]
    phasen=np.linspace(min(phasei)*1.05,max(phasei)*1.05,600)
    kguess=dpcy(nguessdict[night])
    kguess[rpi]=1e-6
    kf,kslope,_,_,_=fitmodel(phasei,kguess)
    if kslope==None:
        kslope=0.
    else:
        None 
    fittedmodel=fitmodel(phasei,kguess)[0]+phasei*kslope
    # Keplerian corrected joint data
    ax1.errorbar(phasei,rvi,np.array(joint_unsigrv),fmt="k.", ecolor="k", capsize=1.,ms=0,capthick=0.5,elinewidth=0.7,zorder=9)
    ax1.errorbar(phasei,rvi,np.array(joint_sigrv),fmt="b.", ecolor="green", capsize=2.,ms=2,capthick=0.5,elinewidth=0.7,zorder=10)
    ###
    fmodel,_,sigw,_,_=dpcy(fitmodel(phasen,nguessdict[night]))
    zlist=sorted(zip(phasei,rvi,sigrvi))
    phasei,rvi,sigrvi=np.array(zlist).T
    m=upfit(phasen,nguessdict[night],phasei,rvi,sigrvi)
    rvfitk=upfitlin(phasen,kguess,_,_,_)

    for k in range(nsamples):
        try: 
            ax1.plot(phasen,1000*(upfit(phasen,sampledir[k][night],phasei,rvi,sigrvi)-rvfitk), color="gray", alpha=0.1)
        except:
            None
    ax1.plot(phasen,1000*(m-rvfitk),color="r",linewidth=0.5)
    ####
# Plots the transit ingress and egress
    ax1.axvspan(-tr_dur/2.-tr_dict["dT0"], tr_dur/2.-tr_dict["dT0"], alpha=0.3, color='orange')
    ax1.axvspan(-tr_ingress_egress/2.-tr_dict["dT0"], tr_ingress_egress/2.-tr_dict["dT0"], alpha=0.4, color='orange')
    ax1.set_ylabel("RV [m/s]")
    ax1.set_xlabel("Phase")
    ax1.set_xlim([min(phasei)*1.05,max(phasei)*1.05])
    plt.savefig("Joint_fit_"+wbin+"_Keplerian_corrected.pdf", transparent=True,bbox_inches='tight')
    plt.close()
    #############################################################################################################
    if gps==True:
        plt.style.use('seaborn-paper')
        fig = plt.figure(figsize=(5,3))
        ax1=fig.add_subplot()
        ax1.errorbar(joint_phase,joint_rv_res2,joint_unsigrv,fmt="k.", ecolor="k", capsize=1.,ms=0,capthick=0.5,elinewidth=0.7,zorder=9)
        ax1.errorbar(joint_phase,joint_rv_res2,joint_sigrv,fmt="b.", ecolor="green", capsize=2.,ms=2,capthick=0.5,elinewidth=0.7,zorder=10)
        ax1.set_ylabel("RV [m/s]")
        ax1.set_xlabel("Phase")
        plt.savefig("Joint_residuals_withGPs_"+wbin+".pdf", transparent=True,bbox_inches='tight')
        plt.close()
        
        #fig = plt.figure(figsize=(5,3))
        #ax1=fig.add_subplot()


    else:
        None


fig=plt.figure(figsize=(6,4), dpi= 100, facecolor='w', edgecolor='k')


wavelengths=[ast.literal_eval(k) for k in sdict["INFO"]["BINS"]]
mean_wave=[(k[0]+k[1])/2 for k in wavelengths]
wave_bin=np.array([k[1] for k in wavelengths]) - np.array(mean_wave)
#print(wavelengths)
#print(mean_wave)
#print(wave_bin)
estimated_params=np.array(parestsave).T
rp=[k[nights][0] for k in estimated_params]
rpmax=[k[nights][1] for k in estimated_params]
rpmin=[k[nights][2] for k in estimated_params]
print(rp,rpmax,rpmin)
plt.style.use('seaborn-paper')

plt.errorbar(mean_wave[1:],np.array(rp[1:]),yerr=[rpmin[1:],rpmax[1:]],xerr=wave_bin[1:], fmt="k.",capsize=3.,linewidth=1.0)
plt.xlabel("Wavelength [nm]")
plt.ylabel(r"$R_P/ R_*$")

plt.savefig("Transmission.pdf", transparent=True,bbox_inches='tight')

f = open("out_transmission.txt", "w+")
f.write("Central_Wavelength"+";"+"Wavelength_Interval" +";"+"Rp"+";"+"Rp_error_u"+";"+"Rp_error_d"+"\n")
for k in range(len(mean_wave)-1):
    k+=1
    f.write(str(mean_wave[k])+";"+str(wave_bin[k])+";"+
            str(rp[k])+";"+str(rpmax[k])+";"+str(rpmin[k])+"\n")
f.close()




plt.close()
#print(mean_wave[1:])
#print(np.array(estimated_params[0]).T)
#fig=plt.figure(figsize=(6,4), dpi= 100, facecolor='w', edgecolor='k')
P=sdict["INFO"]['MODEL_STELLAR_PAR']["P"]
tt=sdict["INFO"]['MODEL_STELLAR_PAR']["delta_transit"]
s1=[]
s2=[]
s3=[]
for k in range(len(estimated_params)-1):
    k+=1
    s1+=[[np.array(estimated_params[k]).T[0][-1]*P*1000.*0.14,np.array(estimated_params[k]).T[1][-1]*P*1000.*0.14]]
    s2+=[[np.array(estimated_params[k]).T[0][-2]*P*1000.*0.14,np.array(estimated_params[k]).T[1][-2]*P*1000.*0.14]]
    #s3+=[[np.array(estimated_params[k]).T[0][-3]*P*1000.*0.14,np.array(estimated_params[k]).T[1][-3]*P*1000.*0.14]]

plt.errorbar(mean_wave[1:],np.array(s1).T[0],yerr=np.array(s1).T[1],capsize=3.,linewidth=1.0,label="Night 1")
plt.errorbar(mean_wave[1:],np.array(s2).T[0],yerr=np.array(s2).T[1],capsize=3.,linewidth=1.0,label="Night 2")

plt.xlabel("Wavelength [nm]")
plt.ylabel(r"$Chromatic slope\, [(m/s)/ day] $")
plt.legend()

plt.savefig("Slope_variation.pdf", transparent=True,bbox_inches='tight')

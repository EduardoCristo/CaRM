import numpy as np
import sys
import matplotlib.pyplot as plt
from scripts.constants import *
import json
from ldtk import LDPSetCreator, BoxcarFilter
import os
from scripts.constants import storbitpar, Wnburn, Wnprod, Cnburn, Cnprod, rvs_paths, model
from astropy.io import fits
from scripts.rm_emcee_call import fit
from scripts.mcmc import mcmc
import pickle


number_nights=len(rvs_paths)


def bjdtophase(bjd):
    # Time of observation MJD
    obs_time = bjd

    # Number of orbits since transit center epoch
    # tepoch=storbitpar["tepoch"]
    norb = (obs_time-storbitpar["tepoch"])/storbitpar["P"]

    # Number of full orbits
    nforb = [round(x) for x in norb]
    phase = norb-nforb
    return(phase)
    

nlbdint=[]
PHASE=[]
RV=[]
SIGRV=[]
c=0
for k in rvs_paths:
    for wint in range(len(k)):
        wavelength_interval=[float(x) for x in ((open(k[wint]).readline()).replace("#","")).split(",")]
        nlbdint+=[wavelength_interval]
        bjd,rv,sigrv=np.loadtxt(k[wint],skiprows=2,delimiter=",").T
        PHASE+=[bjdtophase(bjd)]
        RV+=[rv]
        SIGRV+=[sigrv]
    c+=1

datalen=len(PHASE)


lbdint=nlbdint[0:int(datalen/number_nights)]


#########
cmc = False
while cmc == False:
    try:
        filters = [BoxcarFilter(str(k), lbdint[k][0], lbdint[k][1]) for k in range(len(lbdint))]
        #print([(str(k),lbdint[k][0],lbdint[k][1]) for k in xrange(len(lbdint))])
        sc = LDPSetCreator(teff=(storbitpar["steff"], storbitpar["sig_steff"]),    # Define your star, and the code
                           # downloads the uncached stellar
                           logg=(storbitpar["slog"],
                                 storbitpar["sig_slog"]),
                           # spectra from the Husser et al.
                           z=(storbitpar["sz"], storbitpar["sig_sz"]),
                           filters=filters, cache=os.path.abspath(os.path.join("yourpath", os.pardir))+"/cache/")
        ps = sc.create_profiles()      	         # Create the limb darkening profiles
        if storbitpar["ld_law"]=="ln":
            ldcn, qe = ps.coeffs_ln(do_mc=True)
        if storbitpar["ld_law"]=="qd":
            ldcn, qe = ps.coeffs_qd(do_mc=True)
        if storbitpar["ld_law"]=="nl":
            ldcn, qe = ps.coeffs_nl(do_mc=True)
        else:
            None
    except ValueError:
        print("Ill try to compute again")
    else:
        print("LD parameters computed")
        cmc = True
ldc = ldcn
#########################

print(("The limb darkening coef. are:", ldcn))

# Save data
import pickle
savefile=open("data.pkl","wb")

savedict=dict()
savedict["INFO"]={}
savedict["INFO"]["MODEL_STELLAR_PAR"]=storbitpar
savedict["INFO"]["BINS"]=[str(k) for k in lbdint]
savedict["INFO"]["NUMBER_OBS"]=number_nights
savedict["INFO"]["NUMBER_WHITE_CHAINS"]=Wnchains
savedict["INFO"]["NUMBER_CHROM_CHAINS"]=Cnchains
savedict["INFO"]["NUMBER_WHITE_MCMC_BURN_STEPS"]=Wnburn
savedict["INFO"]["NUMBER_WHITE_MCMC_PROD_STEPS"]=Wnprod
savedict["INFO"]["NUMBER_CHROM_MCMC_BURN_STEPS"]=Cnburn
savedict["INFO"]["NUMBER_CHROM_MCMC_PROD_STEPS"]=Cnprod
savedict["INFO"]["PARAMETERS_GUESS"]=Wguess
savedict["INFO"]["PARAMETERS_WHITE_PRIORS"]=Wpriors
savedict["INFO"]["PARAMETERS_CHROM_PRIORS"]=Cpriors
savedict["INFO"]["PARAMETERS_WHITE_PRIORS_TYPE"]=Wpar
savedict["INFO"]["PARAMETERS_CHROM_PRIORS_TYPE"]=Cpar
savedict["INFO"]["PARAMETERS_WHITE_PAR_EST"]=Wsingle
savedict["INFO"]["PARAMETERS_CHROM_PAR_EST"]=Csingle

c=0
for k in savedict["INFO"]["BINS"]:
    savedict["RUNS"]={}
    savedict["RUNS"][str(k)]={}
    savedict["RUNS"][str(k)]["LIMB_DARKENING"]=[ldc[c],qe[c]]
    c+=1

savedict["INFO"]["LIMB_DARKENING"]=[[ldc[k],qe[k]] for k in range(len(lbdint))]


pickle.dump(savedict, savefile)
savefile.close()

# Fit the white-light
whiterange=[x for x in range(0,datalen,len(lbdint))]
print("Running MCMC for "+savedict["INFO"]["BINS"][0])
updated_par=fit([PHASE[x] for x in whiterange],[RV[x] for x in whiterange],[SIGRV[x] for x in whiterange],ldcn[0],qe[0],0,savedict["INFO"]["BINS"][0])

if dow==1:
    sys.exit("Finished")

ldcoef,lderr=[ldc,qe]


for k in range(len(ldc)-1):
    k+=1
    ldkng,ldkng_err=[ldcoef[k],lderr[k]]
    chromatic_range=[x for x in range(k,datalen,len(lbdint))]
    print("Running MCMC for "+savedict["INFO"]["BINS"][k])
    for j in range(number_nights):
        if storbitpar["ld_law"]=="ln" and storbitpar["model"]=="pyastronomy":
            updated_par[j]["ldc"]=ldkng[0]
            Cpriors["ldc"]=[ldkng[0],ldkng_err[0]]

        elif storbitpar["ld_law"]=="ln" and storbitpar["model"]=="pyarome":
            updated_par[j]["ldc1"]=ldkng[0]
            Cpriors["ldc"]=[ldkng[0],ldkng_err[0]]

        elif storbitpar["ld_law"]=="qd":
            updated_par[j]["ldc1"],updated_par[j]["ldc2"]=ldkng
            Cpriors["ldc1"],Cpriors["ldc2"]=[[ldkng[k],ldkng_err[k]] for k in range(len(ldkng))] 

        elif storbitpar["ld_law"]=="nl":
            Cpriors["ldc1"],Cpriors["ldc2"],Cpriors["ldc3"],Cpriors["ldc4"]=[[ldkng[k],ldkng_err[k]] for k in range(len(ldkng))]
            updated_par[j]["ldc1"],updated_par[j]["ldc2"],updated_par[j]["ldc3"],updated_par[j]["ldc4"]=ldkng
    print("Limb-darkening coefficients: \n")
    print(ldkng,ldkng_err)
    print("Initial Guess: \n")
    print(updated_par)
    from scripts.constants import storbitpar, spect, Cpar, Cpriors,Csingle
    mcmc([RV[int(x)] for x in chromatic_range],[PHASE[int(x)] for x in chromatic_range],[SIGRV[int(x)] for x in chromatic_range],updated_par,Cpar,Cpriors,Csingle,1,savedict["INFO"]["BINS"][k])


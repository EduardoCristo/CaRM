from .variables import modelvar
import numpy as np

rvs_paths=[["./hd189733b_txt/night_1at_377.96_683.62.txt","./hd189733b_txt/night_1at_377.96_421.46.txt","./hd189733b_txt/night_1at_419.65_469.71.txt","./hd189733b_txt/night_1at_468.06_521.44.txt","./hd189733b_txt/night_1at_520.05_569.71.txt","./hd189733b_txt/night_1at_568.63_627.83.txt","./hd189733b_txt/night_1at_633.77_683.62.txt"],["./hd189733b_txt/night_2at_377.96_683.62.txt","./hd189733b_txt/night_2at_377.96_421.46.txt","./hd189733b_txt/night_2at_419.65_469.71.txt","./hd189733b_txt/night_2at_468.06_521.44.txt","./hd189733b_txt/night_2at_520.05_569.71.txt","./hd189733b_txt/night_2at_568.63_627.83.txt","./hd189733b_txt/night_2at_633.77_683.62.txt"],["./hd189733b_txt/night_3at_377.96_683.62.txt","./hd189733b_txt/night_3at_377.96_421.46.txt","./hd189733b_txt/night_3at_419.65_469.71.txt","./hd189733b_txt/night_3at_468.06_521.44.txt","./hd189733b_txt/night_3at_520.05_569.71.txt","./hd189733b_txt/night_3at_568.63_627.83.txt","./hd189733b_txt/night_3at_633.77_683.62.txt"]]

pathlist = []
ldc = []
spect="HARPS"
# Optional
# Do only white? 1-yes, 0-no
dow = 0
# Data saved in a txt file?1-yes, 0-no
dtxt = 0

#GPs?
gps=False
#Slope
useall=True
# Guess from?
parmaxprob=True

# number of iterations on MCMC (niter no burnin e 2xniter em seguida)
Wnburn = 750
Wnprod = 3000
Wnchains = 50

Cnburn = 750
Cnprod = 3000
Cnchains = 50

ncores = 50

#######	Astronomical constants	#######
#Constants (Astronomical_Constants_2017 in data_files)
au = 149597870700  # m
Rsun = 696000000  # m
Rj = 69911000  # m


model = "pyarome"
storbitpar, Wpar, Wguess, Wpriors, Cpar, Cguess, Cpriors, Wsingle, Csingle, vardict_plot = modelvar(
    model)
####### Stellar, transit and orbital parameters ########
storbitpar["model"] = model
storbitpar["Rstar"] = 0.766*Rsun
storbitpar["ld_law"] = "qd"
storbitpar["steff"] = 4969
storbitpar["sig_steff"] = 43
storbitpar["slog"] = 4.60
storbitpar["sig_slog"] = 0.01
storbitpar["sz"] = -0.07
storbitpar["sig_sz"] = 0.02
storbitpar["tepoch"] =2453988.80339
storbitpar["P"] = 	2.21857312
storbitpar["delta_transit"] = 0.07527


######## Initial guess #########
import numpy as np
# Pyarome
####### Model parameters #######
Wguess["vsys"] = -1.9
Wguess["rp"] = 0.1581
Wguess["k"] = 0.20196
Wguess["sma"] = 8.756
Wguess["inc"] = 85.508
Wguess["lda"] = -0.85
Wguess["ldc1"] = None
Wguess["ldc2"] = None
Wguess["beta0"] = 1.3
Wguess["Vsini"] = 3.05
Wguess["sigma0"] = 3.136
Wguess["zeta"] = 4.0
Wguess["Kmax"] = 0.
Wguess["dT0"] = 0.
Wguess["sigw"] = np.log(1/1000.)
Wguess["act_slope"] =0

Wguess["ln_a"] = 0.
Wguess["ln_tau"] = 0.

####### Fit Parameters: (0-default)no (1)yes #######
Wpar["vsys"] = "U"
Wpar["rp"] = "G"
Wpar["Vsini"] = "G"
Wpar["k"] = "G"
Wpar["beta0"] = "G"
Wpar["sigw"] = "U"


##########################
Cpar["vsys"] = "U"
Cpar["rp"] = "G"
Cpar["k"] = "G"
Cpar["sigw"] = "U"




####### Parameters priors (default) [] #######
Wpriors["vsys"] = [-2.2, -1.6]
Wpriors["rp"] = [Wguess["rp"],0.0005]
Wpriors["Vsini"] = [Wguess["Vsini"],0.1]
Wpriors["beta0"] = [Wguess["beta0"],0.1]
Wpriors["k"] = [Wguess["k"],.1]
Wpriors["sigw"] = [-16,np.log(20/1000)]

##########################################
Cpriors["vsys"] = [-2.2, -1.7]
Cpriors["rp"] = [Wguess["rp"], 0.1*Wguess["rp"]]
Cpriors["k"] = [Wguess["k"],.1]
Cpriors["sigw"] = [-16,np.log(50/1000)]


# Parameters we let free between both transits:

Wsingle["vsys"] = True
Wsingle["k"] = True
#Wsingle["act_slope"] = True
Wsingle["sigw"] = True


Csingle["vsys"] = True
Csingle["k"] = True
Csingle["sigw"] = True

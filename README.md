# CaRM
CaRM- Python implementation of the Chromatic Rossiter-McLaughlin effect to retrieve broadband transmission spectra of transiting exoplanets

![Alt text](carm.png?raw=true "Title")
# Chromatic Rossiter–McLaughlin

CaRM is a software written in Python3.8 to compute, using a Markov chain Monte
Carlo (MCMC) algorithm, the radial velocity curves including the RM effect and retrieve
the transmission spectrum of a target. Broadly, there is a constants.py file were the
user gives the input of the properties of the system. From
there to the final output the code can behave automatically as a black box. The green
boxes represent the two main processes to complete the user inputs, compute the RVs
from the CCF files and organize the data. The MCMC algorithm will fit the
RM curves and update the wavelength independent parameters. The code builds a data.pkl file where it is saved, bin by bin sequentially, the data generated from the processes in the software to allow the visualization and retrieval of the results with
the auxiliary reading code.

![Alt text](CARM_flowchart.png?raw=true "Title")

# The models
To fit the Rossiter-McLaughlin anomaly, CaRM incorporates two models: ARoME (Boué et al. 2013) and Ohta et al. (2005) (implemented in PyAstronomy, Czesla et al. 2019). The parameters of each model can be accessed with the following keys:

### PyAstronomy RM model
Vrot- Stellar rotation velocity [km/s];  
Is- Inclination of stellar rotation axis [degrees];
Omega- Angular rotation velocity (star) [rad/s];

### ARoME
beta0- Width of the non-rotating star [km/s];  
Vsini- Projected stellar rotation velocity [km/s];  
sigma0- Width of the best Gaussian fit to the CCFs [km/s]
zeta- Macro-turbulence amplitude [km/s]

#### Adittional (common to all models)
vsys- Systematic velocity of the system [km/s];
rp- Radius of the transiting planet [R_*];  
k- Keplerian semi-amplitude [km/s];  
sma- Semi-major axis [R_*];  
inc- Orbital inclination [degrees];  
lda- Spin-orbit angle [degrees];  
dT0- Mid transit time shift (phase);  
sigw- Log jitter amplitude;  
act_slope- Linear slope [km/s];  
ln_a- Log-amplitude of the GP kernel;  
ln_tau- Log-timescale of the GP kernel;  

# How to run it?
In the current version you need to have a CaRM copy for each run. First make a copy to the folder where you will run it. Next all the input is made changing the values of the "constants.py" file. For that it is needed to provide the radial velocity data, optionally in the following formats:

1) Folders with HARPS or ESPRESSO CCF files:
main_path=[[folder_night1],[folder_night2],...];

2) Text files with RVs:
rvs_paths=[[night1_bin1.txt,night1_bin2.txt,...],[night2_bin1.txt,night2_bin2.txt,...],...].


If you go with the text files RVs option, you will need to provide them in the following format:

#Initial_wavelength, final_wavelength [nm]

#Observing time [BJD],RV[km/s],RV_ERR[km/s]

Comma separated data

.

.

.


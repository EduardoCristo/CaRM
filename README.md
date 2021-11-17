# CaRM
CaRM- Python implementation of the Chromatic Rossiter-McLaughlin effect to retrieve broadband transmission spectra of transiting exoplanets

![Alt text](carm.png?raw=true "Title")
# Chromatic Rossiter–McLaughlin

CaRM is a software written in Python(2.xx and 3.xx) to compute, using a Markov chain Monte
Carlo (MCMC) algorithm, the radial velocity curves including the RM effect and retrieve
the transmission spectrum of a target. Broadly there is a constants.py file were the
user gives the input of the properties of the system. From
there to the final output the code can behave automatically as black box. The green
boxes represent the two main processes to complete the user inputs, compute the RVs
from the CCF files and organize the data. The MCMC algorithm will fit the
RM curves and update the wavelength independent parameters. When the code finishes
results.fits (the output) file will contain all the data generated from the processes
in the software to allow the visualization and retrieval of the results, white boxes, with
the auxiliary reading code.

![Alt text](CARM_flowchart.png?raw=true "Title")

# How to run it?
How to run the code?
1) Edit constants.py to specify the initial parameters, the absolute path to the data, stellar and orbital parameters.

2) Run orderfit.py in the shell :  python2.x orderfit.py 
   OR
   Run orderfit.py in the shell :  python3.x orderfit.py 
   OR
   Run orderfit_slurm.sh to run it in supernova: sbatch orderfit_slurm.sh

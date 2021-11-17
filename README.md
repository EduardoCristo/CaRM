# CaRM
CaRM- Python implementation of the Chromatic Rossiter-McLaughlin effect to retrieve broadband transmission spectra of transiting exoplanets

![Alt text](carm.png?raw=true "Title")
# Chromatic Rossiter–McLaughlin

CaRM is a software written in Python(2.xx and 3.xx) to compute, using a Markov chain Monte
Carlo (MCMC) algorithm, the radial velocity curves including the RM effect and retrieve
the transmission spectrum of a target. Broadly there is a constants.py file were the
user gives the input of the properties of the system. From
there to the final output the code can behave automatically as a black box. The green
boxes represent the two main processes to complete the user inputs, compute the RVs
from the CCF files and organize the data. The MCMC algorithm will fit the
RM curves and update the wavelength independent parameters. When the code finishes
results.fits (the output) file will contain all the data generated from the processes
in the software to allow the visualization and retrieval of the results, white boxes, with
the auxiliary reading code.

![Alt text](CARM_flowchart.png?raw=true "Title")

# How to run it?
In the current version you need to have a CaRM copy for each run. First make a copy to the folder where you will run it. Next all the input is made changing the values of the "constants.py" file. For that it is needed to provide the radial velocity data, optionally in the following formats:

1) Folders with HARPS or ESPRESSO CCF files:
rvs_paths=[[folder_night1],[folder_night2],...]

2) Text files with RVs:
rvs_paths=[[night1_bin1.txt,night1_bin2.txt,...],[night2_bin1.txt,night2_bin2.txt,...],...]
import numpy as np
from .mcmc import mcmc
import scripts.constants as ct
import scipy.stats as st
import scipy.optimize as optimization
import sys
import os
import matplotlib.pyplot as plt
from .aromefit import fitmodel, kepler
from copy import deepcopy as dpcy


def parcalc(phase, rv, sigrv, pguess, dtype=1):
    Mc = mcmc(rv, phase, pguess, sigrv, dtype)
    return(Mc)

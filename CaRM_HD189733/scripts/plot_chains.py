from matplotlib.pyplot import subplots, figure, Subplot, Axes  # , figure, plot, show
import numpy as np
from collections import OrderedDict  # defaultdict


def plot_chains(chains, lnprobability, l_param_name=None, l_walker=None, l_burnin=None,
                suppress_burnin=False, plot_height=2, plot_width=8, truths=None, **kwargs_tl):
    ndim = chains.shape[-1]
    nwalker = chains.shape[0]
    fig, ax = subplots(nrows=ndim + 1, sharex=True, squeeze=True,
                       figsize=(plot_width, ndim * plot_height))
    #l_walker = __get_default_l_walker(l_walker=l_walker, nwalker=nwalker)
    #l_param_name = __get_default_l_param_name(l_param_name=l_param_name, ndim=ndim)
    #l_burnin = __get_default_l_burnin(l_burnin=l_burnin, nwalker=nwalker)

    l_walker = np.arange(0, nwalker, 1)
    #l_param_name = __get_default_l_param_name(l_param_name=l_param_name, ndim=ndim)
    l_burnin = np.zeros(nwalker)

    lnprob_min = lnprobability[l_walker, ...].min()
    lnprob_max = lnprobability[l_walker, ...].max()
    p = 0
    for walker, burnin in zip(l_walker, l_burnin):
        ax[0].set_title("lnpost")
        line = ax[0].plot(lnprobability[walker, :], alpha=0.5)
        ax[0].vlines(burnin, lnprob_min, lnprob_max, color=line[0].get_color(), linestyles="dashed",
                     alpha=0.5)
    for i in range(ndim):
        ax[i + 1].set_title(l_param_name[i])
        vmin = chains[l_walker, :, i].min()
        vmax = chains[l_walker, :, i].max()
        for walker, burnin in zip(l_walker, l_burnin):
            if suppress_burnin:
                line = ax[i + 1].plot(chains[walker, burnin:, i], alpha=0.5)
            else:
                line = ax[i + 1].plot(chains[walker, :, i], alpha=0.5)
                ax[i + 1].vlines(burnin, vmin, vmax, color=line[0].get_color(), linestyles="dashed",
                                 alpha=0.5)
                try:
                    if truths != None:
                        ax[i+1].axhline(truths[i], color="k")
                    else:
                        None
                except:
                    None
    ax[ndim].set_xlabel("iteration")
    fig.tight_layout(**kwargs_tl)

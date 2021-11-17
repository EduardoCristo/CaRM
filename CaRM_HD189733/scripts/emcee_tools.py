#!/usr/bin/python
# -*- coding:  utf-8 -*-
"""
emcee tools module.

The objective of this module is to provide a toolbox for the exploitation and visualisation of emcee
results.
"""
from logging import getLogger, INFO
from matplotlib.pyplot import subplots, figure, Subplot, Axes  # , figure, plot, show
import numpy as np
from numpy import linspace, median, where, array, argmax, unravel_index, ones, nan, sqrt, argsort
from numpy import percentile, exp, newaxis, concatenate, std, isfinite, delete
from numbers import Number
# from sys import stdout
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
# from copy import deepcopy
from collections import OrderedDict  # defaultdict
from tqdm import tqdm
#from PyAstronomy.pyasl import foldAt
from dill import dump, load
from os import makedirs, getcwd
from os.path import isfile, join
from pandas import read_table
# import pprint

#from .stats.loc_scale_estimator import mad
#from .tqdm_logger import TqdmToLogger
#from .time_series_toolbox import get_time_supersampled, average_supersampled_values
#from .human_machine_interface.QCM import QCM_utilisateur
#from ..posterior.core.likelihood.jitter_noise_model import jitter_name
#from ..posterior.core.likelihood.manager_noise_model import Manager_NoiseModel
# from ..posterior.core.posterior import alldtst_key


from scipy.stats import mode

# from ipdb import set_trace

"""
## Logger Object
logger = getLogger(__name__)

mgr_noisemodel = Manager_NoiseModel()
mgr_noisemodel.load_setup()


exptime_Kepler = 0.02043402778  # days
"""


def get_centre_gaussian(xdata, ydata):
    """Return the centre of a guassian
    :param xdata and ydata are the x and y that define the guassian
        they can be the x and y of an histogram
    return double that is the centre of the guassian
    """
    from lmfit.models import GaussianModel
    gmodel = GaussianModel()
    params = gmodel.make_params(
        amplitude=ydata.max(), center=xdata.mean(), sigma=xdata.std())
    result = gmodel.fit(ydata, params, x=xdata)
    peak = result.values['center']
    return peak


def gauspeak(values, nbins):
    """Return the centre of a guassian fit to an histrogram of values
    :param values np.array with values for which we will calculate the histogram and the centre of a gaussianfit
        if np.array has more than one parameter it will do each one seperatly
    :param nbins int number of bins used in the histrogram
    return list of the guassian centre fit for each parameter
    """
    number_fitted = len(values[0, :])
    peak = []
    for i in range(0, number_fitted):
        ydata = np.histogram(values[:, i], nbins)[0]
        bin_edges = np.histogram(values[:, i], nbins)[1]
        delta = bin_edges[2] - bin_edges[1]
        xdata = bin_edges[0:len(bin_edges) - 1] + delta / 2.
        centre_gaussian = get_centre_gaussian(xdata, ydata)
        centre_gaussian = max(centre_gaussian, xdata[0])
        centre_gaussian = min(centre_gaussian, xdata[nbins - 1])
        peak.append(centre_gaussian)

    return peak


def modepeak(values, nbins):
    """Return the mode a distribution by cumputing the histogram
    :param values np.array with values for which we will calculate the mode of the distribution
        if np.array has more than one parameter it will do each one seperatly
    :param nbins int number of bins used in the histrogram
    return list of the mode fit for each parameter
    """
    number_fitted = len(values[0, :])
    peak = []
    for i in range(0, number_fitted):
        ydata = np.histogram(values[:, i], nbins)[0]
        bin_edges = np.histogram(values[:, i], nbins)[1]
        delta = bin_edges[2] - bin_edges[1]
        xdata = bin_edges[0:len(bin_edges) - 1] + delta / 2.
        indice = np.argmax(ydata)
        peak.append(xdata[indice])

    return peak


def get_init_distrib_from_fitvalues(fitted_values):
    """Generate the init_distrib dictionary for generate_random_init_pos from fitted_values.
    :param pd.DataFrame fitted_values: Fitted values from a previous run rows are parameter names
        columns are value, sigma+, sigma-

    :return dict init_distrib: dictionary of dictionary specifying the parameters "mu" and "sigma"
        of the normal distribution to use for each parameter. First level keys are parameter full
        name. Second is "sigma" and "mu".
    """
    init_distrib = {}
    for param, row in fitted_values.iterrows():
        init_distrib[param] = {"mu": row["value"],
                               "sigma": np.mean([row["sigma+"], row["sigma-"]])}
    return init_distrib


def generate_random_init_pos(nwalker, post_instance, init_distrib=None):
    """Generate initial position from for the walkers.

    :param int nwalker: number of walkers
    :param Posterior post_instance: Instance of the posterior class
    :param dict init_distrib: dictionary of dictionary specifying the parameters "mu" and "sigma" of
        the normal distribution to use for each parameter. First level keys are parameter full name.
        Second is "sigma" and "mu".

    :return np.ndarray p0: Ndarray containing the initial positions for all the walkers
    """
    l_param_name = post_instance.lnposteriors.dataset_db["all"].arg_list["param"]
    p0 = []
    if init_distrib is None:
        return post_instance.model.get_initial_values(list_paramnames=l_param_name, nb_values=nwalker).transpose()
    else:
        for param in l_param_name:
            if param in init_distrib:
                p0.append(np.random.normal(loc=init_distrib[param]["mu"],
                                           scale=init_distrib[param]["sigma"],
                                           size=nwalker))
            else:
                p0.append(np.squeeze(np.asarray([post_instance.model.
                                                 get_initial_values(
                                                     list_paramnames=[param])
                                                 for i in range(nwalker)])))
        return np.asarray(p0).transpose()


def explore(sampler, p0, nsteps, save_to_file=False, filename_chain="chain.dat",
            filename_acceptfrac="acceptfrac.dat", dat_folder=None, overwrite=None, l_param_name=None,
            logger=None):
    """Perform an emcee exploration.

    :param emcee.EnsembleSampler sampler: EnsembleSampler instance
    :param array p0: Initial position for each walker and each parameter
    :param bool save_to_file: If True the status of the chains are stored at each iteration in .dat files
    :param str filename_chain: File name to use to save the chains (if save_to_file is True)
    :param str filename_acceptfrac: File name to use to save the acceptance fraction of the chains (if save_to_file is True)
    :param str dat_folder: Folder where the chain and acceptance fraction dat file will be (if save_to_file is True)
    :param bool overwrite: If True already existing .dat files with the same names are automatically overwritten
    :param list_of_str l_param_name: List of the parameter names
    """
    if save_to_file:
        makedirs(dat_folder, exist_ok=True)
        file_chain = join(dat_folder, filename_chain)
        file_acceptfrac = join(dat_folder, filename_acceptfrac)
        for filename, cat in [(file_chain, "chain"), (file_acceptfrac, "acceptfrac")]:
            if isfile(filename):
                if overwrite is None:
                    l_reponses_possibles = ["y", "n"]
                    question = "File {} already exists. Do you want to continue and overwrite it ? {}\n".format(
                        filename, l_reponses_possibles)
                    rep = QCM_utilisateur(question, l_reponses_possibles)
                    overwrite = (rep == "y")
            else:
                overwrite = True
            if overwrite:
                if cat == "chain":
                    with open(filename, "w") as f:
                        f.write("i_walker\t{:s}\n".format(
                            "\t".join(l_param_name + ["lnposterior", ])))
            else:
                raise ValueError(
                    "{} correspond to an existing file.".format(filename))
    if logger is None:
        tqdm_out = None
    else:
        tqdm_out = TqdmToLogger(logger, level=INFO)
    with tqdm(total=nsteps, file=tqdm_out) as pbar:
        previous_i = -1
        for i, result in enumerate(sampler.sample(p0, iterations=nsteps, storechain=True)):
            position = result[0]
            lnprob = result[1]
            if save_to_file:
                with open(filename_chain, "a") as f:
                    for k in range(position.shape[0]):
                        f.write("{:4d} {:s} {:>16.14g}\n".format(k, " ".join(
                            ["{:>16.14g}".format(xx) for xx in position[k]]), lnprob[k]))
                acceptance_fraction = sampler.acceptance_fraction
                with open(filename_acceptfrac, "w") as f:
                    for k, acceptfrac in enumerate(acceptance_fraction):
                        f.write("{:4d} {:>15f}\n".format(k, acceptfrac))
            pbar.update(i - previous_i)
            previous_i = i
        return result


def read_chaindatfile(chaindatfile, walker_col="i_walker", lnpost_col="lnposterior"):
    """Read .dat file created by the explore function (save_to_file=True)

    The .dat file needs to have a header. The fist column has to be i_walker giving the index of the
    walker. The last column has to be lnposterior giving the log posterior probability

    :param str chaindatfile: Path to .dat file
    :param str walker_col: Name of the column containing the index of the walkers
    :param str lnpost_col: Name of the column containing the log posterior probability values
    :return array chains: Array containing the chains formatted as the EnsembleSampler object
    :return array lnpost: Array containing the lnposterior values formatted as the EnsembleSampler
        object
    :return list_of_str l_param: Array containing the lnposterior values formatted as the EnsembleSampler
        object
    """
    df = read_table(chaindatfile, sep="\s+", header=0)

    nb_walker = df[walker_col].max() - df[walker_col].min() + 1
    df["iteration"] = np.array(df.index) // 88
    df.set_index([walker_col, 'iteration'], inplace=True)
    l_param = list(df.columns)
    l_param.remove(lnpost_col)
    return (concatenate([df.loc[walker, :][df.columns[:-1]].values[newaxis, ...] for walker in range(nb_walker)]),
            concatenate([df.loc[walker, :][lnpost_col].values[newaxis, ...]
                         for walker in range(nb_walker)]),
            l_param
            )


def read_acceptfracdatfile(acceptfracdatfile, walker_col="i_walker", lnpost_col="lnposterior"):
    """Read .dat file created by the explore function (save_to_file=True)

    The .dat file needs to have a header. The fist column has to be i_walker giving the index of the
    walker. The last column has to be lnposterior giving the log posterior probability

    :param str acceptfracdatfile: Path to .dat file
    :return array acceptance_fraction: Array containing the acceptance fraction for each walker.
    """
    df = read_table(acceptfracdatfile, sep="\s+", header=0)

    return df[df.columns[-1]].values


def plot_chains(chains, lnprobability, l_param_name=None, l_walker=None, l_burnin=None,
                suppress_burnin=False, plot_height=2, plot_width=8, **kwargs_tl):
    ndim = chains.shape[-1]
    nwalker = chains.shape[0]
    fig, ax = subplots(nrows=ndim + 1, sharex=True, squeeze=True,
                       figsize=(plot_width, ndim * plot_height))
    l_walker = __get_default_l_walker(l_walker=l_walker, nwalker=nwalker)
    l_param_name = __get_default_l_param_name(
        l_param_name=l_param_name, ndim=ndim)
    l_burnin = __get_default_l_burnin(l_burnin=l_burnin, nwalker=nwalker)
    lnprob_min = lnprobability[l_walker, ...].min()
    lnprob_max = lnprobability[l_walker, ...].max()
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
    ax[ndim].set_xlabel("iteration")
    fig.tight_layout(**kwargs_tl)


def overplot_one_data_model(param, l_param_name, datasim, dataset, datasim_kwargs={}, model_instance=None,
                            oversamp=10, supersamp_model=1, exptime=exptime_Kepler,
                            phasefold=False, phasefold_kwargs=None, datasim_dbf_instmod=None,
                            zoom=None, show_title=True, show_legend=True, ax_data=None, ax_resi=None):
    """Zoom on the data model overplot for one datasetself.

    :param np.array param: Vector of parameter values for the model
    :param list_of_string l_param_name: List of parameter name corresponding to the parameter values
        provided in param
    :param DatasimDocFunc datasim: Datasimulator for the dataset.
    :param Dataset dataset: Dataset
    :param Core_Model model_instance: Core_Model instance
    :param int oversamp: The model will be computed in oversamp times more points than the data
    :param int supersamp_model: Each point in which the model is compute will be supersampled by the number
                                of points provided, meaning that we will actually compute the model at
                                supersamp_model points spread over the exposure time (exptime) and then
                                average over this points.
    :param float exptime: exposure time for the supersampling
    :param bool phasefold: If true the phase folded data and model are plotted accord to the ephemeris
        provided in phasefold_kwargs.
    :param dict phasefold_kwargs: Kwargs for the phase folded plot with 3 parameters
        "planet" giving the planet name (string)
        "P" giving the planet orbital period (float)
        "tc" giving the time of inferior conjunction for the planet (float)
    :param dict datasim_dbf_instmod: Database containing the datasim function per planet. For the folded
        plot, we use the models for each planet contribution to be able to display the model and data
        correspond only to the planet whose ephemeris is used to phase fold
        (datasim_dbf.instrument_db[inst_mod_fullname]).
    :param None/list_of_float zoom: If provided the plot will be zoomed. Meaning that the model and data
        will only be plotted between two abscisse values. It should be a list-like object with two elements.
        zoom[0] give the minimum abscisse value for the zoom and zoom[1] give the maximum. If phasefold
        is true the abscisse values are interpreted ass orbital phases, if not as times.
        You also have the possibility to produce several zooms. In this case, zoom should be an array
        or list of list  object where zoom[i][0] is the min abscisse value and zoom[i][1] the max.
    :param bool show_title: If True, show the title giving the dataset name.
    :param bool show_legend: If True, show the legend.
    :param ~matplotlib.axes._axes.Axes ax_data: Axes instance where the data and model will be ploted
    :param ~matplotlib.axes._axes.Axes ax_resi: Axes instance where the residuals will be ploted
    """
    # Ensure that zoom has the good format
    if zoom is None:
        zoom = [[None, None], ]
        nb_plots = 1
    elif isinstance(zoom[0], Number):
        zoom = [zoom, ]
        nb_plots = 1
    else:
        nb_plots = np.shape(zoom)[0]
    # Create ax_data and/or ax_resi if not provided
    if (ax_data is None) and (ax_resi is None):
        fig, axes = subplots(nrows=2, ncols=nb_plots, squeeze=False)
        ax_data = axes[0]
        ax_resi = axes[1]
    elif ax_data is None:
        fig, ax_data = subplots(ncols=nb_plots, squeeze=False)
        ax_data = ax_data[0]
    elif ax_resi is None:
        fig, ax_resi = subplots(ncols=nb_plots, squeeze=False)
        ax_resi = ax_resi[0]
    else:
        if isinstance(ax_data, Axes):
            ax_data = [ax_data, ]
        if isinstance(ax_resi, Axes):
            ax_resi = [ax_resi, ]
    # Initialise title
    title = "{}".format(dataset.dataset_name)
    # Get the instrument model object and the noise model object
    inst_mod = model_instance.get_instmod(dataset.dataset_name)
    noise_mod = mgr_noisemodel.get_noisemodel_subclass(inst_mod.noise_model)
    # Get data point (time, data, data_err) and other kwargs
    kwargs = dataset.get_kwargs()
    t = kwargs.pop("t")
    nt = len(t)
    data = kwargs.pop("data")
    data_err = kwargs.pop("data_err")
    kwargs.update(datasim_kwargs)
    # Extract the jitter information:
    # jitter which give the value of the jitter (float)
    # jitter_type which give the type of jitter model used (string: 'multi' or 'add')
    if noise_mod.has_jitter:
        jitter_param_fullname = inst_mod.parameters[jitter_name].get_name(
            include_prefix=True, recursive=True)
        if inst_mod.parameters[jitter_name].free:
            idx_jitter = l_param_name.index(jitter_param_fullname)
            jitter = param[idx_jitter]
        else:
            jitter = inst_mod.parameters[jitter_name].value
        jitter_type = noise_mod.jitter_type
    else:
        jitter = None
        jitter_type = None
    # Apply jitter if needed
    data_err_new = data_err if jitter is None else apply_jitter(
        data_err, jitter, jitter_type)
    # Case of phase folding
    if phasefold:
        # Get the planet name, period and time of inferior conjunction from phasefold_kwargs
        planet_name = phasefold_kwargs["planet"]
        P = phasefold_kwargs["P"]
        tc = phasefold_kwargs["tc"]
        # Add planet name to title
        title += "pl {}".format(planet_name)
        # Get the datasim for this planet only
        datasim_docfunc_pl = datasim_dbf_instmod[planet_name]
        # Get the datasims for the other planets
        l_datasim_db_docfunc_others = []
        for pl in list(model_instance.planets.keys()):
            if pl == planet_name:
                continue
            else:
                l_datasim_db_docfunc_others.append(datasim_dbf_instmod[pl])
        # Compute the data - other planets contributions
        data_pl = data.copy()
        for datasim_db in l_datasim_db_docfunc_others:
            model, modelwGP, _ = compute_model(t, datasim_db, param, l_param_name,
                                               datasim_kwargs=kwargs,
                                               supersamp=supersamp_model, exptime=exptime,
                                               noise_model=noise_mod,
                                               model_instance=model_instance)
            data_pl = data_pl - model
        # Plot these data phase folded at the ephemeris of the planet.
        pl_kwargs = {"color": "b", "fmt": "."}
        # Plot the model
        for zoom_i, ax_data_i, ax_resi_i in zip(zoom, ax_data, ax_resi):
            _, phases = plot_phase_folded_timeserie(t=t, data=data_pl, P=P, tc=tc, data_err=data_err_new,
                                                    jitter=None, jitter_type=None, zoom=zoom_i, ax=ax_data_i,
                                                    pl_kwargs=pl_kwargs)
            phasemin = phases.min() if zoom_i[0] is None else max(
                [phases.min(), zoom_i[0]])
            phasemax = phases.max() if zoom_i[1] is None else min(
                [phases.max(), zoom_i[1]])
            tmin = tc + P * phasemin
            tmax = tc + P * phasemax
            plot_model(tmin, tmax, nt * oversamp, datasim_docfunc_pl, param, l_param_name,
                       supersamp=supersamp_model, exptime=exptime,
                       datasim_kwargs={'tref': tmin}, plot_phase=True, P=P, tc=tc,
                       noise_model=noise_mod,
                       model_instance=model_instance,
                       ax=ax_data_i)
            # Plot the residuals
            plot_residuals(t, data, datasim_docfunc_pl, param, l_param_name,
                           data_err=data_err_new, jitter=None, jitter_type=None,
                           supersamp=supersamp_model, exptime=exptime,
                           datasim_kwargs=kwargs, plot_phase=True, P=P, tc=tc,
                           noise_model=noise_mod,
                           model_instance=model_instance,
                           ax=ax_resi_i)
    # Case of NOT phase folding
    else:
        for zoom_i, ax_data_i, ax_resi_i in zip(zoom, ax_data, ax_resi):
            # Perform the zoom if needed
            if (zoom_i[0] is not None) and (zoom_i[1] is not None):
                zoomed_arrays, idx_zoom = apply_zoom(
                    zoom=zoom_i, base_array=t, arrays=[data, data_err_new])
                t_i = zoomed_arrays[0]
                data_i = zoomed_arrays[1]
                data_err_new_i = zoomed_arrays[2]
            else:
                t_i = t
                data_i = data
                data_err_new_i = data_err_new
            # plot the data
            ax_data_i.errorbar(t_i, data_i, data_err_new_i, fmt=".", color="b")
            # Plot the model
            tmin = t_i.min()
            tmax = t_i.max()
            plot_model(tmin, tmax, nt * oversamp, datasim, param, l_param_name,
                       datasim_kwargs=kwargs, supersamp=supersamp_model, exptime=exptime,
                       plot_phase=False, noise_model=noise_mod,
                       model_instance=model_instance, ax=ax_data_i)
            # Plot the residuals
            plot_residuals(t_i, data_i, datasim, param, l_param_name, data_err=data_err_new_i,
                           jitter=None, jitter_type=None,
                           datasim_kwargs=kwargs, supersamp=supersamp_model, exptime=exptime,
                           plot_phase=False, noise_model=noise_mod,
                           model_instance=model_instance, ax=ax_resi_i)
    # Print the title if required
    if show_title:
        ax_data[0].set_title(title)
    # Plot the legend
    if show_legend:
        ax_data[0].legend(loc='upper right', shadow=True)


def overplot_data_model(param, l_param_name, datasim_dbf, dataset_db, datasim_kwargs={},
                        model_instance=None, oversamp=10, supersamp_model=1, exptime=exptime_Kepler,
                        phasefold=False, phasefold_kwargs=None,
                        plot_height=2, plot_width=8, kwargs_tl={}):
    """Overplot datasets and model for each dataset and provide the residuals.

    WARNING: In its current status (using the new overplot_one_data_model function). This function has
    not been tested with phasefold=True.

    :param np.array param: Vector of parameter values for the model
    :param list_of_string l_param_name: List of parameter name corresponding to the parameter values
        provided in param
    :param datasim_dbf: Datasimulator database
    :param DatasetDatabase dataset_db:
    :param Core_Model model_instance: Core_Model instance
    :param int oversamp: The model will be computed in oversamp times more points than the data
    :param int supersamp_model: Each point in which the model is compute will be supersampled by the number
                                of points provided, meaning that we will actually compute the model at
                                supersamp_model points spread over the exposure time (exptime) and then
                                average over this points.
    :param float exptime: exposure time for the supersampling
    :param bool phasefold: If true the phase folded data and model are plotted.
    :param dict phasefold_kwargs: Kwargs for the phase folded plot with 3 parameters:
        "planets" giving the planet names (list)
        "P" giving the planet periods (list)
        "tc" giving the times of inferior conjunction for each planet (list)
    :param plot_height:
    :param plot_width:
    :param kwargs_tl:
    """
    # Check that if phasefold is True phasefold_kwargs is not None
    if phasefold and (phasefold_kwargs is None):
        raise ValueError(
            "If you want to phase fold, you have to provide the phasefold_kwargs")

    # Get the list of all datasets names and the number of datasets
    l_datasets = dataset_db.get_datasets()
    ndataset = len(l_datasets)

    # Create the figure and grid which will harbor the plots for each dataset
    fig = figure(figsize=(plot_width, ndataset * plot_height))
    gs = GridSpec(nrows=ndataset, ncols=1)

    for ii, dataset in enumerate(l_datasets):
        inst_mod_fullname = model_instance.get_instmod_fullname(
            dataset.dataset_name)
        datasim = datasim_dbf.instrument_db[inst_mod_fullname]["whole"]
        datasim_dbf_instmod = datasim_dbf.instrument_db[inst_mod_fullname]
        if phasefold:
            # Create two sets of axes to produce a phase-folded dataset+model  and the residuals plot
            # for each planet
            nplanet = 1 if phasefold_kwargs is None else len(
                phasefold_kwargs["planets"])
            (axes_data,
             axes_resi) = add_twoaxeswithsharex_perplanet(gs[ii],
                                                          nplanet=nplanet,
                                                          fig=fig,
                                                          gs_from_sps_kw={"height_ratios": (3, 1)})
            # Produce the phase-folded plots for each planet
            for planet_name, P, tc, ax_data, ax_resi in zip(phasefold_kwargs["planets"],
                                                            phasefold_kwargs["P"],
                                                            phasefold_kwargs["tc"],
                                                            axes_data, axes_resi):
                overplot_one_data_model(param=param, l_param_name=l_param_name, datasim=datasim, dataset=dataset,
                                        datasim_kwargs=datasim_kwargs, model_instance=model_instance,
                                        oversamp=oversamp, supersamp_model=supersamp_model, exptime=exptime,
                                        phasefold=phasefold, phasefold_kwargs={
                                            "planet": planet_name, "P": P, "tc": tc},
                                        datasim_dbf_instmod=datasim_dbf_instmod, zoom=None, show_title=True,
                                        show_legend=True, ax_data=ax_data, ax_resi=ax_resi)
        else:
            # Create the two axes data+model and residuals
            ax_data, ax_resi = add_twoaxeswithsharex(
                gs[ii], fig=fig, gs_from_sps_kw={"height_ratios": (3, 1)})
            # Produce the plots
            overplot_one_data_model(param=param, l_param_name=l_param_name, datasim=datasim, dataset=dataset,
                                    datasim_kwargs=datasim_kwargs, model_instance=model_instance,
                                    oversamp=oversamp, supersamp_model=supersamp_model, exptime=exptime,
                                    phasefold=phasefold,
                                    datasim_dbf_instmod=datasim_dbf_instmod, zoom=None, show_title=True,
                                    show_legend=True, ax_data=ax_data, ax_resi=ax_resi)
    fig.tight_layout(**kwargs_tl)


def overplot_onedata_model_pertransits(P, t_tr, planet_name, param, l_param_name, datasim, dataset,
                                       datasim_kwargs={}, model_instance=None,
                                       oversamp=10, supersamp_model=1, exptime=exptime_Kepler,
                                       zoom_width=0.25, show_title=True, show_legend=True,
                                       plot_height=2, plot_width=2, kwargs_tl={}):
    """Zoom on the data model overplot for one datasetself.

    :param float/list_of_float P: Orbital period(s)
    :param float/list_of_float t_tr: Transit time(s), t0(s) of the ephemeris
    :param string/list_of_string planet_name: Planet name(s)
    :param np.array param: Vector of parameter values for the model
    :param list_of_string l_param_name: List of parameter name corresponding to the parameter values
        provided in param
    :param DatasimDocFunc datasim: Datasimulator for the dataset.
    :param Dataset dataset: Dataset
    :param Core_Model model_instance: Core_Model instance
    :param int oversamp: The model will be computed in oversamp times more points than the data
    :param int supersamp_model: Each point in which the model is compute will be supersampled by the number
                                of points provided, meaning that we will actually compute the model at
                                supersamp_model points spread over the exposure time (exptime) and then
                                average over this points.
    :param float exptime: exposure time for the supersampling
    :param float zoom_width: Width of the zoom.
    :param bool show_title: If True, show the title giving the dataset name.
    :param bool show_legend: If True, show the legend.
    """
    # Get the min and max time of the dataset.
    t = dataset.get_time()
    t_max = max(t)
    t_min = min(t)
    # Define the zooms for each planet ephemeris
    zoom_planets = []
    P = [P, ] if isinstance(P, Number) else P
    t_tr = [t_tr, ] if isinstance(t_tr, Number) else t_tr
    planet_name = [planet_name, ] if isinstance(
        planet_name, str) else planet_name
    nb_pl = len(P)
    for jj, P_pl, t_tr_pl in zip(list(range(nb_pl)), P, t_tr):
        zoom_planets.append([])
        nb_per_min = int((t_min - t_tr_pl) // P_pl)
        nb_per_min = nb_per_min if nb_per_min > 0 else nb_per_min - 1
        nb_per_max = int((t_max - t_tr_pl) // P_pl)
        nb_per_max = nb_per_max if nb_per_max < 0 else nb_per_max + 1
        for ii in range(nb_per_min, nb_per_max + 1):
            t_tr_ii = t_tr_pl + P_pl * ii
            zoom_min_ii = t_tr_ii - zoom_width / 2.
            zoom_max_ii = t_tr_ii + zoom_width / 2.
            if (zoom_max_ii < t_min) or (zoom_min_ii > t_max):
                continue
            else:
                zoom_planets[jj].append([zoom_min_ii, zoom_max_ii])
    # Defines the fig and axes
    nb_max_zoom = max([len(zoom_pl) for zoom_pl in zoom_planets])
    fig, axes = subplots(nrows=2 * nb_pl, ncols=nb_max_zoom,
                         figsize=(plot_width * nb_max_zoom, nb_pl * plot_height))
    for jj in range(nb_pl):
        overplot_one_data_model(param=param, l_param_name=l_param_name, datasim=datasim, dataset=dataset,
                                datasim_kwargs=datasim_kwargs, model_instance=model_instance, oversamp=oversamp,
                                supersamp_model=supersamp_model, exptime=exptime_Kepler, phasefold=False,
                                zoom=zoom_planets[jj], show_title=False, show_legend=False, ax_data=axes[jj * 2],
                                ax_resi=axes[jj * 2 + 1])
    fig.tight_layout(**kwargs_tl)


def compute_model(t, datasim_db_docfunc, param, l_param_name, datasim_kwargs=None,
                  supersamp=1, exptime=exptime_Kepler,
                  noise_model=None, model_instance=None):
    # Supersample the time if needed
    if supersamp > 1:
        t_model = get_time_supersampled(t, supersamp, exptime)
    else:
        t_model = t

    # If datasim_kwargs is None affect an empty dict and no additional arguments will be passed to
    # the datasim function
    if datasim_kwargs is None:
        datasim_kwargs = {}

    # Compute the model values for each time
    idx_par = []
    datasim_function = datasim_db_docfunc.function
    datasim_paramnames = datasim_db_docfunc.params_model
    for par in datasim_paramnames:
        idx_par.append(l_param_name.index(par))
    model = datasim_function(param[idx_par], t_model, **datasim_kwargs)
    if supersamp > 1:
        model = average_supersampled_values(model, supersamp)

    # Compute GP contribution if needed.
    if noise_model is not None:
        if noise_model.has_GP:
            # WARNING: Here for GP model. I take all the instruments with the same GP noise
            # model to compute the GP contribution. FOr now it works but If at one point I want
            # to use the same model for two different time of data (ex: RV and LC). It will not anymore.
            # Create the simulated data (model) for all the datasets and get the datasim_all
            datasim_all = (model_instance.
                           create_datasimulator_alldatasets(dataset_db=model_instance.dataset_db))
            idx_datasim = []
            for param_name in datasim_all.params_model:
                idx_datasim.append(l_param_name.index(param_name))
            model_all = datasim_all.function(param[idx_datasim])
            print(("model_all: {}".format(model_all)))
            # Get the list of instrument models which have the same GP noise model that the Current
            # Dataset you try to model
            l_instmod_noisemod_cat = []
            for inst_mod in model_instance.get_instmodels_used():
                if inst_mod.noise_model == noise_model.category:
                    l_instmod_noisemod_cat.append(inst_mod.get_name(
                        include_prefix=True, recursive=True))
            # Get the simulated data (model) of only the dataset with the current GP noisemodel
            # Get the dataset kwargs mandatory for the GP simulation (l_datakwargs_noisemod)
            l_datakwargs_noisemod = []
            if datasim_all.multi_output:
                idx_noisemod_GP = [ii for ii, instmod_fullname in
                                   enumerate(datasim_all.instmodel_fullname)
                                   if instmod_fullname in l_instmod_noisemod_cat]
                l_dataset_noisemod_cat = datasim_all.dataset.iloc[idx_noisemod_GP]
                model_noisemodel_GP = [mod_ii for ii, mod_ii in enumerate(model_all)
                                       if ii in idx_noisemod_GP]
                for dataset_name in l_dataset_noisemod_cat:
                    dataset = model_instance.dataset_db[dataset_name]
                    l_datakwargs_noisemod.append(
                        noise_model.get_necessary_datakwargs(dataset))
            else:
                model_noisemodel_GP = [model_all]
                dataset_name = datasim_all.dataset.iloc[0]
                dataset = model_instance.dataset_db[dataset_name]
                l_datakwargs_noisemod.append(
                    noise_model.get_necessary_datakwargs(dataset))
            print(("model: {}".format(model_noisemodel_GP)))
            print(("l_datakwargs: {}".format(l_datakwargs_noisemod)))
            # Get the GP simulator for the current GP noise model
            (gpsim_func,
             l_param_noisemod) = noise_model.get_gp_simulator(model_instance, l_param_name)
            # Get the value of the parameter to provide to the GP simulator.
            # WARNING: If one of the gp simulator parmameter is fixed it might not get it with the code
            # below
            idx_noisemod = []
            for param_name in l_param_noisemod:
                idx_noisemod.append(l_param_name.index(param_name))
            # Compute the GP model
            gp_model = gpsim_func(model_noisemodel_GP, param[idx_noisemod],
                                  l_datakwargs_noisemod,
                                  t_model)
            if supersamp > 1:
                gp_model = np.mean(gp_model.reshape(-1, supersamp), axis=1)
            model_wGP = model + gp_model
        else:
            model_wGP = None
    else:
        model_wGP = None

    return model, model_wGP, t_model


def plot_model(tmin, tmax, nt, datasim_db_docfunc, param, l_param_name, datasim_kwargs=None,
               supersamp=1, exptime=exptime_Kepler,
               plot_phase=False, P=None, tc=None,
               noise_model=None, model_instance=None,
               pl_kwargs_model=None, pl_kwargs_modelandGP=None,
               ax=None):
    """
    """
    # Create the time sampling (tsamp) and the tmin and tmax for the model computation (tmin_moins,
    # tmax_plus), the model time vector (t)
    # nt - 1 because this the number of intervals
    tsamp = (tmax - tmin) / (nt - 1)
    tmin_moins = tmin - tsamp  # Add 1 point before tmin
    tmax_plus = tmax + tsamp  # Add 1 point after tmax
    nt += 2
    t_plot = linspace(tmin_moins, tmax_plus, nt)
    model, model_wGP, t = compute_model(t_plot, datasim_db_docfunc, param, l_param_name,
                                        datasim_kwargs=datasim_kwargs, supersamp=1,
                                        exptime=exptime_Kepler,
                                        noise_model=noise_model,
                                        model_instance=model_instance)

    # Create a new figure and ax if needed
    ax = __get_default_ax(ax=ax)

    # Plot the model
    kwarg_model = {"label": "model", "color": "g", "fmt": "-", "alpha": 0.6}
    if pl_kwargs_model is not None:
        kwarg_model.update(pl_kwargs_model)
    if plot_phase:
        line, _ = plot_phase_folded_timeserie(
            t_plot, model, P, tc, ax=ax, pl_kwargs=kwarg_model)
    else:
        line = ax.errorbar(t_plot, model, **kwarg_model)

    # Plot the model + GP
    if model_wGP is not None:
        kwarg_GP = {"label": "model+GP",
                    "color": "r", "fmt": "-", "alpha": 0.6}
        if pl_kwargs_modelandGP is not None:
            kwarg_GP.update(pl_kwargs_modelandGP)
        if plot_phase:
            line_wGP, _ = plot_phase_folded_timeserie(t_plot, model_wGP, P, tc, ax=ax,
                                                      pl_kwargs=kwarg_GP)
        else:
            line_wGP = ax.errorbar(t_plot, model_wGP, **kwarg_GP)
    else:
        line_wGP = None
    return line, line_wGP


def plot_residuals(t, data, datasim_db_docfunc, param, l_param_name,
                   datasim_kwargs=None, data_err=None, jitter=None, jitter_type=None,
                   supersamp=1, exptime=exptime_Kepler, plot_phase=False, P=None, tc=None,
                   noise_model=None, model_instance=None, zoom=None,
                   pl_kwargs_model=None, show_model=True,
                   pl_kwargs_modelandGP=None, show_modelandGP=True,
                   ax=None):
    """
    :param array t:
    :param array data:
    :param datasim_db_docfunc:
    :param param:
    :param l_param_name:
    :param datasim_kwargs:
    :param data_err:
    :param jitter:
    :param jitter_type:
    :param supersamp:
    :param exptime:
    :param plot_phase: If True, plot phase folded residuals
    :param P:
    :param tc:
    :param noise_model:
    :param model_instance:
    :param zoom:
    :param pl_kwargs_model:
    :param bool show_model: To show the residuals of the model only when the noise model is a GP.
    :param pl_kwargs_modelandGP:
    :param bool show_modelandGP:
    :param ax:
    """
    # Create a new figure and ax if needed
    ax = __get_default_ax(ax=ax)
    # Compute residual
    model, model_wGP, _ = compute_model(t, datasim_db_docfunc, param, l_param_name,
                                        datasim_kwargs=datasim_kwargs, supersamp=supersamp,
                                        exptime=exptime,
                                        noise_model=noise_model,
                                        model_instance=model_instance)
    residual = data - model
    # Apply jitter if needed
    data_err_new = data_err if jitter is None else apply_jitter(
        data_err, jitter, jitter_type)
    # Determine if noise model is GP
    if noise_model is None:
        noise_modelGP = False
    else:
        noise_modelGP = noise_model.has_GP
    # Perform zoom if needed (I do the zoom after computing the model with all the data because
    # I am not sure that doing the zoom before will not change the result for GP noise model.)
    # I do the zoom only if I don't want to phase fold, because in this case I can do it inside plot_phase_folded_timeserie
    if (zoom is not None) and not(plot_phase):
        extra_arrays_to_zoom = [data, model, residual] if data_err is None else [
            data, model, residual, data_err_new]
        if noise_modelGP:
            extra_arrays_to_zoom.append(model_wGP)
        zoomed_arrays, idx_zoom = apply_zoom(
            zoom=zoom, base_array=t, arrays=extra_arrays_to_zoom)
        t_zoom = zoomed_arrays[0]
        data_zoom = zoomed_arrays[1]
        residual_zoom = zoomed_arrays[2]
        if data_err is None:
            data_err_new_zoom = None
            model_wGP_zoom = None if not(noise_modelGP) else zoomed_arrays[3]
        else:
            data_err_new_zoom = zoomed_arrays[3]
            model_wGP_zoom = None if not(noise_modelGP) else zoomed_arrays[4]
    # Plot the residuals of model only (even if noise model is GP)
    if show_model or not(noise_modelGP):
        kwarg_model = {"label": "model", "color": "g", "fmt": "."}
        if pl_kwargs_model is not None:
            kwarg_model.update(pl_kwargs_model)
        if plot_phase:
            plot_phase_folded_timeserie(t, residual, P, tc, data_err=data_err_new, zoom=zoom, ax=ax,
                                        pl_kwargs=kwarg_model)
        else:
            if zoom is not None:
                ax.errorbar(t_zoom, residual_zoom,
                            data_err_new_zoom, **kwarg_model)
            else:
                ax.errorbar(t, residual, data_err_new, **kwarg_model)
    # Plot the residuals of model + GP
    if (noise_model is not None) and noise_modelGP and show_modelandGP:
        residual_wGP = data - model_wGP if zoom is None else data_zoom - model_wGP_zoom
        kwarg_GP = {"label": "model+GP",
                    "color": "r", "fmt": ".", "alpha": 0.6}
        if pl_kwargs_modelandGP is not None:
            kwarg_GP.update(pl_kwargs_modelandGP)
        if plot_phase:
            plot_phase_folded_timeserie(t, residual_wGP, P, tc, data_err=data_err_new, zoom=zoom, ax=ax,
                                        pl_kwargs=kwarg_GP)
        else:
            if zoom is not None:
                ax.errorbar(t_zoom, residual_wGP,
                            data_err_new_zoom, **kwarg_GP)
            else:
                ax.errorbar(t, residual_wGP, data_err_new, **kwarg_GP)
    # Draw a line y=0 for the residuals
    xmin, xmax = ax.get_xlim()
    ax.hlines(y=0.0, xmin=xmin, xmax=xmax, linestyles="dashed", linewidth=1)
    ax.set_xlim(xmin, xmax)


def apply_jitter(data_err, jitter, jitter_type):
    """Apply jitter to the data error bar

    :param array_float data_err: data error array
    :param float jitter: jitter value
    :param str jitter_type: jitter_type ("multi" or "add")
    """
    # Adapt the data_err to the jitter value is needed.
    if jitter_type == "multi":
        data_err_new = data_err * exp(jitter)
    elif jitter_type == "add":
        data_err_new = sqrt(data_err**2 * (1 + exp(2 * jitter)))
    else:
        raise ValueError("jitter_type should be in ['multi', 'add']")
    return data_err_new


def apply_zoom(zoom, base_array, arrays=None):
    """Apply jitter to the data error bar

    :param list_of_float zoom: It should be a list-like object with two elements.
        zoom[0] give the minimum value in zoom_base_array for the zoom and zoom[1] give the maximum.
    :param array_of_float base_array: Array on which the zoom in based. The idx of the elements
        which satisfy zoom[0] < zoom_base_array < zoom[1], will be used to cut both zoom_base_array
        and zoomed_arrays
    :param None/list_of_array arrays: List of array to zoom.
    :return list_of_array zoomed_arrays: List of zoomed arrays, the first one is the zoomed based array
    :return array idx_zoom: array of indexes which satisfy the zoom
    """
    idx_zoom = where((base_array > zoom[0]) & (base_array < zoom[1]))[0]
    zoomed_arrays = []
    zoomed_arrays.append(base_array[idx_zoom])
    if arrays is not None:
        for arr in arrays:
            zoomed_arrays.append(arr[idx_zoom])
    return zoomed_arrays, idx_zoom


def plot_phase_folded_timeserie(t, data, P, tc, data_err=None, jitter=None, jitter_type=None,
                                zoom=None, ax=None, pl_kwargs=None):
    """Plot a phase folded representation of a lc

    :param array_float t: time array
    :param array_float data: data array
    :param float P: Period of the planet
    :param float tc: Time of inferior conjuction of the planet
    :param array_float data_err: data error array
    :param float jitter: jitter value
    :param str jitter_type: jitter_type ("multi" or "add")
    :param None/list_of_float zoom: If provided the plot will be zoom. Meaning that the model and data
        will only be plotted between two phase values. It should be a list-like object with two elements.
        zoom[0] give the minimum phase for the zoom and zoom[1] give the maximum.
    :param ~matplotlib.axes._axes.Axes ax: Axes instance where the data and model will be ploted
    :param dict pl_kwargs: Keyword argument passed to pl.errorbar function

    P and tc needs to have the same unit than the t
    """
    # Create a new figure and ax if needed
    ax = __get_default_ax(ax=ax)
    # Obtain the phases with respect to some ephemerid P and tc
    phases = foldAt(t, P, T0=(tc + P / 2)) - 0.5
    # Sort with respect to phase
    sortIndi = argsort(phases)
    # If data error provided
    if data_err is not None:
        # Apply jitter if needed
        data_err_new = data_err if jitter is None else apply_jitter(
            data_err, jitter, jitter_type)
        # Create the sorted data_err vector
        data_err_new_sort = data_err_new[sortIndi]
    # Create the sorted phase, sorted data vectors
    phase_sort = phases[sortIndi]
    data_sort = data[sortIndi]
    # Perform zoom if needed
    if (zoom is not None):
        if (zoom[0] is not None) and (zoom[1] is not None):
            extra_arrays_to_zoom = [data_sort] if data_err is None else [
                data_sort, data_err_new_sort]
            zoomed_arrays, idx_zoom = apply_zoom(
                zoom=zoom, base_array=phase_sort, arrays=extra_arrays_to_zoom)
            phase_sort = zoomed_arrays[0]
            data_sort = zoomed_arrays[1]
            data_err_new_sort = None if data_err is None else zoomed_arrays[2]
    # Check the errorbar kwargs
    kw = dict() if pl_kwargs is None else pl_kwargs.copy()
    if "fmt" not in kw:
        kw["fmt"] = "-"
    if "color" not in kw:
        kw["color"] = "r"
    # Plot the phase folded data
    if data_err is not None:
        line = ax.errorbar(phase_sort, data_sort, data_err_new_sort, **kw)
    else:
        line = ax.errorbar(phase_sort, data_sort, **kw)
    return line, phases


def add_twoaxeswithsharex(subplotspec, fig, gs_from_sps_kw=None):
    """Add two axes to a subplotspec (created with gridspec) for data and residual plot. """
    # Set the default values for GridSpecFromSubplotSpec
    kw = dict() if gs_from_sps_kw is None else gs_from_sps_kw.copy()
    if "hspace" not in kw:
        kw["hspace"] = 0.1
    if "height_ratios" not in kw:
        kw["height_ratios"] = (4, 1)

    # Create the two axes, share x axis and set the ticks and ticks label properties
    gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=subplotspec, **kw)
    ax0 = Subplot(fig, gs[0])
    ax0.locator_params(axis="y", tight=True, nbins=4)
    ax0.tick_params(labelbottom="off")
    fig.add_subplot(ax0)
    ax1 = Subplot(fig, gs[1], sharex=ax0)
    fig.add_subplot(ax1)
    ax1.locator_params(axis="y", tight=True, nbins=4)

    # Return the two axes
    return ax0, ax1


def add_twoaxeswithsharex_perplanet(subplotspec, nplanet, fig, gs_from_sps_kw=None):
    """Add two axes per planet to a subplotspec (created with gridspec) for data and residual plot.
    """
    # Create the nplanet axes
    gs = gridspec.GridSpecFromSubplotSpec(1, nplanet, subplot_spec=subplotspec)

    # Create the two axes for the data and the residuals for each planet
    axes_data = []
    axes_resi = []
    for gs_elem in gs:
        ax_data, ax_resi = add_twoaxeswithsharex(
            gs_elem, fig, gs_from_sps_kw=gs_from_sps_kw)
        axes_data.append(ax_data)
        axes_resi.append(ax_resi)
    return axes_data, axes_resi


# Work in Progress to detect outliers before plotting
# def apply_mask(x=None):
#     '''
#     Returns the outlier mask, an array of indices corresponding to the non-outliers.
#
#     :param numpy.ndarray x: If specified, returns the masked version of :py:obj:`x` instead.
#        Default :py:obj:`None`
#
#     WORK IN PROGRESS
#     '''
#
#     if x is None:
#         return np.delete(np.arange(len(self.time)), self.mask)
#     else:
#         return np.delete(x, self.mask, axis=0)


def acceptancefraction_selection(acceptance_fraction, sig_fact=3., quantile=75, verbose=1):
    """Return selected walker based on the acceptance fraction.

    :param np.array acceptance_fraction: Value of the acceptance fraction for each walker.
    :param float sig_fact: acceptance fraction below mean - sig_fact * sigma will be rejected
    :param int verbose: if 1 speaks otherwise not
    """
    percentile_acceptance_frac = percentile(acceptance_fraction, quantile)
    mad_acceptance_frac = mad(acceptance_fraction)
    if verbose == 1:
        logger.info("Acceptance fraction of the walkers: {}\nquantile {}%: {}, MAD:{}"
                    "".format(acceptance_fraction, quantile, percentile_acceptance_frac,
                              mad_acceptance_frac))
    l_selected_walker = where(acceptance_fraction > (percentile_acceptance_frac -
                                                     sig_fact * mad_acceptance_frac))[0]
    nb_rejected = acceptance_fraction.shape[0] - len(l_selected_walker)
    if verbose == 1:
        logger.info("Number of rejected walkers: {}/{}".format(nb_rejected,
                                                               acceptance_fraction.shape[0]))
    return l_selected_walker, nb_rejected


def lnposterior_selection(lnprobability, sig_fact=3., quantile=75, quantile_walker=50, verbose=1):
    """Return selected walker based on the acceptance fraction.

    :param np.array lnprobability: Values of the lnprobability taken by each walker at each iteration
    :param float sig_fact: acceptance fraction below quantile - sig_fact * sigma will be rejected
    :param float quantile: Quantile to use as reference lnprobability value.
    :param float quantile_walker: Quantile used to assert the lnprobability for each walker. 50 is
        the meadian, 100 is the highest lnprobability.
    :param int verbose: if 1 speaks otherwise not
    :return list_of_int l_selected_walker: list of selected walker
    :return int nb_rejected:  number of rejected walker
    """
    walkers_percentile_lnposterior = percentile(
        lnprobability, quantile_walker, axis=1)
    percentile_lnposterior = percentile(
        walkers_percentile_lnposterior, quantile)
    mad_lnposterior = mad(walkers_percentile_lnposterior)
    if verbose == 1:
        logger.info("lnposterior of the walkers: {}\nquantile {}%: {}, MAD:{}"
                    "".format(walkers_percentile_lnposterior, quantile, percentile_lnposterior,
                              mad_lnposterior))
    l_selected_walker = where(walkers_percentile_lnposterior > (
        percentile_lnposterior - (sig_fact * mad_lnposterior)))[0]
    nb_rejected = lnprobability.shape[0] - len(l_selected_walker)
    if verbose == 1:
        logger.info(
            "Number of rejected walkers: {}/{}".format(nb_rejected, lnprobability.shape[0]))
    return l_selected_walker, nb_rejected


def get_fitted_values(chainI, method="MAP", l_param_name=None, l_walker=None, l_burnin=None,
                      lnprobability=None, verbose=1):
    """Return the fitted values from the sampler.

    :param ChainInterpret chainI:
    :param string method: method used to extract the fitted values ["MAP", "median", "gausfit", "mode"]
    :param int_iteratable l_walkers: list of valid walkers
    :param int burnin: index of the first iteration to consider.
    :param int verbose: if 1 speaks otherwise not
    """
    ndim = chainI.dim
    if method == "median":
        res = np.nanmedian(get_clean_flatchain(chainI, l_walker=l_walker, l_burnin=l_burnin),
                           axis=0)
    elif method == "MAP":
        if (l_walker is not None) or (l_burnin is not None):
            logger.warning(
                "With method MAP the l_walker and l_burnin arguments are ignored.")
        walker, it = unravel_index(
            argmax(lnprobability), dims=lnprobability.shape)
        res = array([chainI[walker, it, dim] for dim in range(ndim)])
    elif method == "gaussfit":
        res = gauspeak(get_clean_flatchain(
            chainI, l_walker=l_walker, l_burnin=l_burnin), nbins=100)
    elif method == "mode":
        res = modepeak(get_clean_flatchain(
            chainI, l_walker=l_walker, l_burnin=l_burnin), nbins=100)
    else:
        raise ValueError("Method {} is not recognised".format(method))
    if verbose == 1:
        l_param_names = __get_default_l_param_name(l_param_name, ndim)
        text = "\n"
        for i, param_name in enumerate(l_param_names):
            text += "{} = {}\n".format(param_name, res[i])
        logger.info(text)
    return res


def get_clean_flatchain(chainI, l_walker=None, l_burnin=None, force_finite=True):
    """Return a flatchain with only the selected walkers and iteration after the burnin.

    :param ChainInterpret chainI:
    :param int_iteratable l_walkers: list of valid walkers
    :param int_iteratable l_burnin: list of burnin iterations for each valid walker
    :param bool force_finite: If True the function will suppress every iteration for which one of the
        parameter values provided is not finite.
    :return np.array res: cleaned flat chain
    """
    res = None
    # If no walker list is provided nor burnin list, the result is the whole flatten chain
    if (l_walker is None) and (l_burnin is None):
        res = chainI.flatchain
    # If no then just select the walkers provided by l_walker and return the flat chain
    elif l_burnin is None:
        sh = chainI[l_walker, ...].shape
        res = chainI[l_walker, ...].reshape(sh[0] * sh[1], sh[2])
    # If no walker list is provided get the default list
    elif l_walker is None:
        l_walker = __get_default_l_walker(nwalker=chainI.shape[0])
    # If res is None then at this point we have a l_burnin and a l_walker
    if res is None:
        ndim = chainI.dim
        res = []
        # Case where there is only one free parameter
        if ndim == 1:
            for walker, burnin in zip(l_walker, l_burnin):
                res.extend(chainI[walker, burnin:])
            res = array(res).transpose()
        # Case where there is several free parameter
        else:
            for dim in range(ndim):
                res.append([])
                for walker, burnin in zip(l_walker, l_burnin):
                    res[dim].extend(chainI[walker, burnin:, dim])
            res = array(res).transpose()
    # Remove iteration where one of the parameter is not finite
    if force_finite:
        return np.delete(res, np.where(np.logical_not(np.isfinite(res)))[0], axis=0)
    else:
        return res


def geweke_multi(chains, first=0.1, last=0.5, intervals=20, l_walker=None):
    """Adapted the geweke test for multiple wlaker exploration.

    :param emcee.EnsembleSampler sampler:
    :param float first: first portion of the chain to be used in the Geweke diagnostic.
        Default to 0.1 (i.e. first 10 % of the chain)
    :param float last: last portion of the chain to be used in the Geweke diagnostic.
        Default to 0.5 (i.e. last 50 % of the chain)
    :param int intervals: Number of sub-chains to analyze. Defaults to 20.
    :param int_iterable l_walker: list of valid walkers
    """
    nwalker = chains.shape[0]
    ndim = chains.shape[-1]
    # Get the list of valid walkers (l_walker), the number of parameters (ndim) and the number of
    # steps for each walker (nsteps)
    l_walker = __get_default_l_walker(l_walker=l_walker, nwalker=nwalker)
    nwalker = len(l_walker)
    nsteps = chains.shape[1]

    # Compute the start step of the last part of the chain and compute median and MAD of the last
    # part of the chain for each parameter
    nb_step_last = int(nsteps * last)
    last_start_step = nsteps - nb_step_last
    logger.info(
        "Number of steps in last portion of the chains for geweke convergence estimate: {}".format(nb_step_last))
    l_med_last = [median(chains[l_walker, last_start_step:, dim])
                  for dim in range(ndim)]
    logger.info("Median value for each parameter (over all specified walkers) in the last portion of"
                " the chains: {}".format(l_med_last))
    l_mad_last = [mad(chains[l_walker, last_start_step:, dim])
                  for dim in range(ndim)]
    l_mad_last_is0 = [mad_dim == 0.0 for mad_dim in l_mad_last]
    if any(l_mad_last_is0):
        for dim in np.where(l_mad_last_is0)[0]:
            logger.debug(
                "MAD returned 0.0 for parameter number: {}. Compute std.".format(dim))
            l_mad_last[dim] = std(chains[l_walker, last_start_step:, dim])
            if l_mad_last[dim] == 0.0:
                raise ValueError(
                    "MAD and std returned zero for parameter number: {}.".format(dim))
    logger.info("MAD value for each parameter (over all specified walkers) in the last portion of"
                " the chains: {}".format(l_mad_last))

    # Compute the start steps of all the first parts of the chains that we will use for the Geweke
    # diagnostic (first_start_steps) and length of those first part (first_length).
    first_length = int(nsteps * first)
    logger.info(
        "Number of steps in each interval of first portion of the chains: {}".format(first_length))
    first_start_steps = [int(i * (last_start_step / intervals))
                         for i in range(intervals)]
    logger.debug("First step of each interval in the first portion of the chains: {}".format(
        first_start_steps))

    # Then for each parameter and for each walker and for each first part compute the Geweke z-score
    zscores = ones((nwalker, intervals, ndim)) * nan
    for dim, med_last, mad_last in zip(list(range(ndim)), l_med_last, l_mad_last):
        for i, walker in enumerate(l_walker):
            for j, first_start in enumerate(first_start_steps):
                med_first = median(
                    chains[walker, first_start:(first_start + first_length), dim])
                mad_first = mad(chains[walker, first_start:(
                    first_start + first_length), dim])
                # Compute the zscore, but if the dispersion of the first part is too big compared to
                # the last part, I don't consider the first part as converge what the zscore.
                if mad_first < (5 * mad_last):
                    zscores[i, j, dim] = (
                        med_first - med_last) / (sqrt(mad_first**2 + mad_last**2))
                else:
                    zscores[i, j, dim] = np.inf
    return zscores, first_start_steps


def geweke_plot(zscores, first_steps=None, l_param_name=None, geweke_thres=2,
                plot_height=2, plot_width=8, **kwargs_tl):
    ndim = zscores.shape[-1]
    nwalker = zscores.shape[0]
    fig, ax = subplots(nrows=ndim, sharex=True, squeeze=True,
                       figsize=(plot_width, ndim * plot_height))
    l_param_name = __get_default_l_param_name(
        l_param_name=l_param_name, ndim=ndim)
    first_steps = __get_default_first_steps(
        first_steps=first_steps, intervals=zscores.shape[1])
    xmin = min(first_steps)
    xmax = max(first_steps)
    for i in range(ndim):
        ax[i].set_title(l_param_name[i])
        for k in range(nwalker):
            ax[i].plot(first_steps, zscores[k, :, i], alpha=0.5)

        ax[i].hlines([-geweke_thres, geweke_thres],
                     xmin, xmax, linestyles="dashed")
    ax[ndim - 1].set_xlabel("iteration")
    fig.tight_layout(**kwargs_tl)


def geweke_selection(zscores, first_steps=None, geweke_thres=2., l_walker=None, verbose=1):
    """Compute the burnin for each valid walker based on their zscores.

    :param numpy.ndarray zscores:
    :param int_iteratable l_walker: list of valid walkers
    """
    res = abs(zscores) <= geweke_thres
    nwalker = zscores.shape[0]
    intervals = zscores.shape[1]
    first_steps = __get_default_first_steps(
        first_steps=first_steps, intervals=intervals)
    l_walker = __get_default_l_walker(l_walker=l_walker, nwalker=nwalker)
    l_burnin = []
    l_walker_new = []
    for i in range(nwalker):
        for j in range(intervals):
            if res[i, j:, :].all():
                l_burnin.append(first_steps[j])
                l_walker_new.append(l_walker[i])
                break
    if verbose == 1:
        logger.info("List of burnin for valid walker: {}".format(
            dict(list(zip(l_walker, l_burnin)))))
        logger.info("Number of walkers invalid walkers found: {}/{}"
                    "".format(len(l_walker) - len(l_walker_new), len(l_walker)))
    return l_burnin, l_walker_new


def __get_default_l_walker(l_walker, nwalker):
    if l_walker is None:
        l_walker = [i for i in range(nwalker)]
    return l_walker


def __get_default_l_param_name(l_param_name, ndim):
    if l_param_name is None:
        l_param_name = [str(i) for i in range(ndim)]
    return l_param_name


def __get_default_l_burnin(l_burnin, nwalker):
    if l_burnin is None:
        l_burnin = [0 for i in range(nwalker)]
    return l_burnin


def __get_default_first_steps(first_steps, intervals):
    return __get_default_l_walker(first_steps, intervals)


def __get_default_ax(ax):
    if ax is None:
        fig, ax = subplots(nrows=1, ncols=1, squeeze=True)
    return ax


def write_latex_table(filename, df_fitval, obj_name=None):
    """Write a TeX file with a table giving the fitted values."""
    if obj_name is None:
        obj_name = ''
    # Create Latex table
    with open(filename, "w") as f:
        f.write(
            "\\begin{table}\n\\caption{\\label{}}\n\\begin{tabular}{lc}\\hline\n")
        f.write("Parameters & {}\\\\ \\hline\n".format(obj_name))
        for par in df_fitval.index:
            f.write("{} & ${}_{{-{}}}^{{+{}}}$\\\\\n".format(par.replace('_', '\\_'),
                                                             df_fitval.loc[par,
                                                                           'value'],
                                                             df_fitval.loc[par,
                                                                           'sigma-'],
                                                             df_fitval.loc[par, 'sigma+']))
        f.write("\\hline\n\\\\")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


extension_pickle = {"chain": "_chain.pk",
                    "lnpost": "_lnprobability.pk",
                    "acceptfrac": "_acceptance_fraction.pk",
                    "l_param_name": "_l_param_name.pk",
                    "df_fittedval": "_df_fittedval.pk",
                    "fitted_values": "_fitted_values.pk",
                    "fitted_values_sec": "_fitted_values_sec.pk",
                    }


def pickle_stuff(stuff, filename):
    """Save stuff in a pickle file.

    The pickle file name is defined by the object_name and the extension
    "{}{}".format(obj_name, extension)

    :param stuff: Stuff to pickle
    :param str filename: Name of the pickle file
    """
    # Save chain in a pickle
    with open(filename, "wb") as fpickle:
        dump(stuff, fpickle)


def save_emceesampler(sampler, l_param_name=None, obj_name="", folder=None):
    """Save Emcee EnsembleSampler instance elements into pickle files.

    :param emcee.EnsembleSampler sampler: EnsembleSampler instance to save
    :param list_of_str l_param_name: list of the parameter names
    :param str obj_name: Object name
    :param str folder: Folder where to put the pickle files
    """
    if folder is None:
        folder = getcwd()
    else:
        makedirs(folder, exist_ok=True)

    # Save chain in a pickle
    pickle_stuff(sampler.chain, join(
        folder, "{}{}".format(obj_name, extension_pickle["chain"])))

    # Save lnprobability in a pickle
    pickle_stuff(sampler.lnprobability, join(
        folder, "{}{}".format(obj_name, extension_pickle["lnpost"])))

    # Save acceptance_fraction in a pickle
    pickle_stuff(sampler.acceptance_fraction, join(
        folder, "{}{}".format(obj_name, extension_pickle["acceptfrac"])))

    # Save l_param_name in a pickle
    if l_param_name is not None:
        pickle_stuff(l_param_name, join(folder, "{}{}".format(
            obj_name, extension_pickle["l_param_name"])))


def save_chain_analysis(obj_name, fitted_values=None, fitted_values_sec=None, df_fittedval=None, folder=None):
    """Save chain analysis results.

    TODO: Update to use pickle_stuff
    """
    if folder is None:
        folder = getcwd()
    else:
        makedirs(folder, exist_ok=True)

    # Save df_fittedval in a pickle
    if df_fittedval is not None:
        with open(join(folder, "{}{}".format(obj_name, extension_pickle["df_fittedval"])), "wb") as fdffitval:
            dump(df_fittedval, fdffitval)

    # Save fitted_values in a pickle
    if fitted_values is not None:
        if "array" not in fitted_values or "l_param" not in fitted_values:
            raise ValueError("fitted_values should be a dictionary with 2 keys 'array' and "
                             "'l_param'")
        with open(join(folder, "{}{}".format(obj_name, extension_pickle["fitted_values"])), "wb") as ffitval:
            dump(fitted_values, ffitval)

    # Save fitted_values in a pickle
    if fitted_values_sec is not None:
        if "array" not in fitted_values or "l_param" not in fitted_values:
            raise ValueError("fitted_values should be a dictionary with 2 keys 'array' and "
                             "'l_param'")
        with open(join(folder, "{}{}".format(obj_name, extension_pickle["fitted_values_sec"])), "wb") as ffitvals:
            dump(fitted_values_sec, ffitvals)


def load_emceesampler(obj_name, folder="."):
    """Save Emcee sampler elements.

    :param str obj_name: Name of the object for which you want to load the chain analysis results.
        This is used to infer the names of the pickle files
    :param str folder:
    """
    if folder is None:
        folder = getcwd()

    # Save chain in a pickle
    with open(join(folder, "{}{}".format(obj_name, extension_pickle["chain"])), "rb") as fchain:
        chain = load(fchain)

    # Save lnprobability in a pickle
    with open(join(folder, "{}{}".format(obj_name, extension_pickle["lnpost"])), "rb") as flnprob:
        lnprobability = load(flnprob)

    # Save acceptance_fraction in a pickle
    with open(join(folder, "{}{}".format(obj_name, extension_pickle["acceptfrac"])), "rb") as faccfrac:
        acceptance_fraction = load(faccfrac)

    # Save l_param_name in a pickle
    with open(join(folder, "{}{}".format(obj_name, extension_pickle["l_param_name"])), "rb") as flparam:
        l_param_name = load(flparam)

    return chain, lnprobability, acceptance_fraction, l_param_name


def load_chain_analysis(obj_name, folder=None):
    """Save Emcee sampler elements.

    :param str obj_name: Name of the object for which you want to load the chain analysis results.
        This is used to infer the names of the pickle files
    :param str folder:
    """
    if folder is None:
        folder = getcwd()

    # load df_fittedval from a pickle
    file_df_fittedval = "{}{}".format(
        obj_name, extension_pickle["df_fittedval"])
    if isfile(join(folder, file_df_fittedval)):
        with open(join(folder, file_df_fittedval), "rb") as fdffitval:
            df_fittedval = load(fdffitval)
    else:
        df_fittedval = None

    # Load fitted_values from a pickle
    file_fitted_values = "{}{}".format(
        obj_name, extension_pickle["fitted_values"])
    if isfile(join(folder, file_fitted_values)):
        with open(join(folder, file_fitted_values), "rb") as ffitval:
            fitted_values = load(ffitval)
    else:
        fitted_values = None

    # Load fitted_values_sec from a pickle
    file_fitted_values_sec = "{}{}".format(
        obj_name, extension_pickle["fitted_values_sec"])
    if isfile(join(folder, file_fitted_values_sec)):
        with open(join(folder, file_fitted_values_sec), "rb") as ffitvals:
            fitted_values_sec = load(ffitvals)
    else:
        fitted_values_sec = None

    return fitted_values, fitted_values_sec, df_fittedval


def get_param_value_OrderedDict(values, l_param_names):
    """Return an Orderedictwith associate the parameter name to its value.
    """
    res = OrderedDict()
    for val, name in zip(values, l_param_names):
        res[name] = val
    return res

from numpy import percentile, exp, newaxis, concatenate, std, isfinite, delete
from numpy import linspace, median, where, array, argmax, unravel_index, ones, nan, sqrt, argsort
from astropy.stats import median_absolute_deviation as mad


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
        print(("lnposterior of the walkers: {}\nquantile {}%: {}, MAD:{}"
              "".format(walkers_percentile_lnposterior, quantile, percentile_lnposterior,
                        mad_lnposterior)))
        """
        logger.info("lnposterior of the walkers: {}\nquantile {}%: {}, MAD:{}"
                    "".format(walkers_percentile_lnposterior, quantile, percentile_lnposterior,
                              mad_lnposterior))
        """
    l_selected_walker = where(walkers_percentile_lnposterior > (
        percentile_lnposterior - (sig_fact * mad_lnposterior)))[0]
    nb_rejected = lnprobability.shape[0] - len(l_selected_walker)
    if verbose == 1:
        print(("Number of rejected walkers: {}/{}".format(nb_rejected,
                                                         lnprobability.shape[0])))
        #logger.info("Number of rejected walkers: {}/{}".format(nb_rejected, lnprobability.shape[0]))
    return l_selected_walker, nb_rejected

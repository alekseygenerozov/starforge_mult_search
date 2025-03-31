from random import choices

import numpy as np
from scipy.optimize import minimize


def pow_simp(g, mm, mmin, mmax):
    """
    Power law pdf

    :param g:  power law index * (-1)
    :param m:  Abscissa
    :param mmin: Inner truncation of power law
    :param mmax: Outer truncation of power law
    """
    return (1 - g) * (mm) ** (-g) / (mmax ** (1 - g) - mmin ** (1 - g))


def nlogProbBin(g, *args):
    """
    Negative log of likelihood

    :param g:  power law index * (-1)
    :param args: Other arguments to pdf (data, minimum and maximum)
    """
    aobs, mmin, mmax = args
    pp = -np.sum([np.log(pow_simp(g, mm, mmin, mmax)) for mm in aobs])
    return pp


def fit_power(aobs, g0, mmin=None, mmax=None):
    """
    Maximum likelihood fit and uncertainty with bootstrap method.

    :param aobs: Observed data
    :param g0: Initial guess for the power law index * (-1)
    :param mmin (None): Inner truncation of power law (If None use minimum of data)
    :param mmax (None): Outer truncation of power law (If None use maximum of data)
    """
    if not mmin:
        mmin = np.min(aobs)
    if not mmax:
        mmax = np.max(aobs)

    soln = minimize(nlogProbBin, g0, args=(tuple(aobs), mmin, mmax))
    gbest = soln["x"][0]
    normbest = (1 - gbest) / (mmax ** (1 - gbest) - mmin ** (1 - gbest))

    return gbest, normbest


def bootstrap(aobs, g0, mmin=None, mmax=None, n=100):
    """
    Maximum likelihood fit and uncertainty with bootstrap method.

    :param aobs: Observed data
    :param g0: Initial guess for the power law index * (-1)
    :param mmin (None): Inner truncation of power law (If None use minimum of data)
    :param mmax (None): Outer truncation of power law (If None use maximum of data)
    :param n (100): Number of trials for bootstrap
    """
    if not mmin:
        mmin = np.min(aobs)
    if not mmax:
        mmax = np.max(aobs)

    fits_all = []
    for ii in range(n):
        re_samp = choices(aobs, k=len(aobs))
        gbest, normbest = fit_power(re_samp, g0, mmin, mmax)
        fits_all.append(gbest)
    return np.mean(fits_all), np.std(fits_all)

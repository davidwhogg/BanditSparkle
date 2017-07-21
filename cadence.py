import numpy as np
from astropy.table import Table
from dateutil import parser as dateutil

__all__ = ["tess", "gaia", "kepler", "k2_short_cadence", "k2_long_cadence"]

def tess():
    return np.linspace(0, 30 * 24 * 60, 1440)


def gaia(path):
    """
    Return an array of Gaia timestamps provided by the Gaia forecasting tool.
    
    :param path:
        The local path containing forecasted observation times from Gaia.
    """

    # Get the forecasted observation times from Gaia for this star
    gaia_obs_time_column = "ObservationTimeAtGaia[UTC]"
    gaia_forecast = Table.read(path, format="csv")

    # Predict the magnitude changes at the observed (delta) times.
    first = dateutil.parse(gaia_forecast[gaia_obs_time_column][0])
    return np.array([(dateutil.parse(v) - first).total_seconds() \
            for v in gaia_forecast[gaia_obs_time_column]])


def kepler():
    r"""
    Return an array of observation times (in seconds) for Kepler long 
    cadence of 30 minute observations over 4 years.
    """
    return np.arange(0, 4 * 365 * 24 * 60 * 60, 30 * 60)


def k2_short_cadence():
    r"""
    Return an array of observing times (in seconds) for K2 short cadence
    observations (1 minute observations for 90 days).
    """
    exp = 60
    return np.arange(0, 90 * 24 * 60 * 60 + exp, exp)


def k2_long_cadence():
    r"""
    Return an array of observing times (in seconds) for K2 long cadence
    observations (30 minute exposures for 90 days).
    """
    exp = 30 * 60
    return np.arange(0, 90 * 24 * 60 * 60 + exp, exp)

import dateutil
import numpy as np
from astropy.table import Table


def tess(n=2500):
    return np.linspace(0, 30 * 24 * 60, n)


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
    first = dateutil.parser.parse(gaia_forecast[gaia_obs_time_column][0])
    return np.array([(dateutil.parser.parse(v) - first).total_seconds() \
            for v in gaia_forecast[gaia_obs_time_column]])

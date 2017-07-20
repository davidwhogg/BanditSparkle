
import itertools
import numpy as np
from astropy.table import Table
from time import time

import cadence
from model import AsteroseismicModel




nu_max, nu_max_error = (129.69e-6, 2.52e-6) # [Hz]
delta_nu, delta_nu_error = (10.93e-6, 0.23e-6) # [Hz]



power_spectrum = Table.read("KIC009145955_d22_v1.fits")


light_curve = Table.read("KIC009145955_d21_v1.fits")
light_curve["TIME"] -= np.min(light_curve["TIME"]) 
light_curve["TIME"] *= 24 * 60.* 60 # convert date tto seconds
light_curve["FLUX"] *= 1e-12 # convert to mags

bell_height = 5e-10

model = AsteroseismicModel()

t = cadence.tess()
y = model.predict_light_curve(t, nu_max, delta_nu, bell_height=bell_height)
omega, ft = model.predict_power_spectrum(t, nu_max, delta_nu,
    bell_height=bell_height)


fig, axes = plt.subplots(3)
axes[0].plot(t, y)
axes[1].semilogx(omega * 1e6, np.abs(ft)**2)
axes[2].semilogx(omega, np.angle(ft, deg=True))

# Plot some data.
axes[1].semilogx(
    power_spectrum["FREQUENCY"], power_spectrum["PSD"],
    c='k', zorder=-1)


# Evaluate log-likelihood for ~5000 Kepler points
N = 1000 # Approximate number of points.
S = int(float(len(light_curve))/N)
t_sparse_kepler = light_curve["TIME"][::S]
y_sparse_kepler = light_curve["FLUX"][::S]
yerr_sparse_kepler = 1e-5 * np.ones(len(t_sparse_kepler))


parameter_names = ("nu_0", "delta_nu", "nu_max", "bell_height", "bell_width", "r_01")

# Estimate bell_width, r_01 and fix it.
grid_bounds = [
    #("nu_0", 100e-6, 260e-6, 10, np.linspace),
    #("delta_nu", 8e-6, 22e-6, 10, np.linspace),
    #("delta_nu", 10.72e-6 - 0.1e-6, 10.72e-6 + 0.35e-6, 1000, np.linspace),
    ("delta_nu", 1.08727027e-05, 1.08727027e-05, 1, np.linspace),
    ("nu_max", 100e-6, 260e-6, 100, np.linspace),
    #("nu_max", nu_max, nu_max, 1, np.linspace),
    #("delta_nu", delta_nu, delta_nu, 1, np.linspace),
    ("nu_0", 129.57e-6, 129.57e-6, 1, np.linspace),
    # By taking max height and near uptake for width
    #("bell_height", 3.43046929e-12,3.43046929e-12, 1, np.linspace),
    #("bell_width", 8.69749003e-09, 8.69749003e-09, 1, np.linspace),
    #("bell_width", -12, 3, 50, np.logspace),
    ("bell_width", 10e-6, 10e-6, 1, np.linspace),
    #("bell_height", -15, -5, 50, np.logspace),
    ("bell_height", 1.0985411419875572e-13, 1.0985411419875572e-13, 1, np.linspace),
    ("r_01", 0.5, 0.5, 1, np.linspace)
]


def get_expected_value(parameter_name):
    if parameter_name in ("nu_0", "nu_max", "delta_nu"):
        return dict(delta_nu=delta_nu).get(parameter_name, nu_max)
    else:
        return np.nan

# Generate thetas.
product_products = [None] * len(parameter_names)
for parameter_name, start, end, P, function in grid_bounds:
    index = parameter_names.index(parameter_name)

    points = function(start, end, P)
    expected_value = get_expected_value(parameter_name)
    #if np.isfinite(expected_value):
    #    points = np.sort(np.hstack([points, expected_value]))

    product_products[index] = points

thetas = np.array([theta for theta in itertools.product(*product_products)])
P = thetas.shape[0]
ll = -np.inf * np.ones(P)

# Set nu_max as nu_0
#thetas[:, 0] = thetas[:, 2]

log_likelihood_cost = 0.75 # seconds per call
print("Expected cost: ~{} seconds ({} theta evaluations @ {} seconds per call"\
    .format(P * log_likelihood_cost, P, log_likelihood_cost))

# Pre-compute required matrices.
model._pre_compute_covariance_matrices(yerr_sparse_kepler)

# Start evaluations.
t_init = time()
for i, theta in enumerate(thetas):
    ll[i] = model.marginalized_log_likelihood(theta, t_sparse_kepler, y_sparse_kepler)

    est_log_likelihood_cost = (time() - t_init)/(i + 1)
    print("{}/{}: {} {} (~{:.0f} secs remaining)".format(
        i, P, theta, ll[i], est_log_likelihood_cost * (P - i + 1)))


# Plot stuff.
scales = dict(nu_0=1e6, nu_max=1e6, delta_nu=1e6, bell_height=1e10)

latex_labels = {
    "nu_max": r"$\nu_{\rm max}$ $[\mu{\rm Hz}]$",
    "nu_0": r"$\nu_{0}$ $[\mu{\rm Hz}]$",
    "delta_nu": r"$\Delta\nu$ $[\mu{\rm Hz}]$",
}
for i, parameter_name in enumerate(parameter_names):

    if np.unique(thetas[:, i]).size == 1:
        print("Skipping {} because it was fixed".format(parameter_name))
        continue

    fig, ax = plt.subplots()

    x = thetas[:, i] * scales.get(parameter_name, 1)

    if parameter_name in ("bell_height", "bell_width"):
        x = np.log10(thetas[:, i])

    ax.scatter(x, ll, 
        alpha=0.5, facecolor="k")
    expected_value = get_expected_value(parameter_name)

    if np.isfinite(expected_value):
        ax.axvline(expected_value * scales.get(parameter_name, 1))

    
    ax.set_xlabel(latex_labels.get(parameter_name, parameter_name))
    ax.set_ylabel(r"$\mathcal{L}$")

raise a

"""

gaia_t = cadence.gaia("KIC12020797_gaia_forecast.csv")
mask = np.zeros(len(light_curve), dtype=bool)

for obs_time in gaia_t:
    index = np.argmin(np.abs(light_curve["TIME"] - obs_time))
    mask[index] = True

gaia_t = light_curve["TIME"][mask]
gaia_flux = light_curve["FLUX"][mask] * 1e-6
#gaia_ivar = light_curve["FLUX_ERR"][mask]**(-2)
gaia_ivar = (20e-6 * np.ones(sum(mask)))**(-2)

theta = [nu_max, delta_nu, nu_max, bell_height, bell_width, r_01]
#foo = model.marginalized_log_likelihood(theta, gaia_t, gaia_flux, gaia_ivar)
"""


nu_maxes = []
ll = []
ll2 = []
points = np.hstack([nu_max, np.linspace(100e-6, 260e-6, 1000)])
#points = np.linspace(nu_max - 10e-6, nu_max + 10e-6, 1000)

nu_max = 160.9e-6 # [Hz]
error_nu_max = 0.5e-6 # [Hz]
delta_nu = 12.89e-6 # [Hz]




theta = [nu_max, delta_nu, nu_max, bell_height, bell_width, r_01]

from time import time
t_a = time()
foo = model.marginalized_log_likelihood(theta, light_curve["TIME"], light_curve["FLUX"], 5e-6)
time_taken = time() - t_a

raise a

N = 1000
for point in points:

    theta = [point, delta_nu, point, bell_height, bell_width, r_01]
    nu_maxes.append(point)
    ll.append(model.marginalized_log_likelihood(theta, light_curve["TIME"][::N], light_curve["FLUX"][::N], (10e-6)**(-2)))
    ll2.append(model.marginalized_log_likelihood(theta, light_curve["TIME"][::N],
        model.predict_light_curve(light_curve["TIME"][::N], nu_max, delta_nu), 1e-6))

# Run a

nu_maxes = np.array(nu_maxes)
ll = np.array(ll)
ll2 = np.array(ll2)

fig, ax = plt.subplots()
ax.scatter(nu_maxes * 1e6, ll)

fig, ax = plt.subplots()
ax.scatter(nu_maxes * 1e6, ll2, facecolor="r")




raise a



axes[0].scatter(
    light_curve["TIME"][mask], light_curve["FLUX"][mask],
    facecolor="k")

t_int = np.linspace(0, t_gaia[-1], 1000)
t_int = light_curve["TIME"][mask]
axes[0].plot(
    t_int, model.predict_light_curve(t_int, nu_max, delta_nu, amplitudes=amplitudes) * 1e6)

raise a





#axes[0].scatter(light_curve["TIME"], light_curve["FLUX"] * 1e6,
#    facecolor="k", zorder=-1)



t_obs = light_curve["TIME"]
flux_obs = light_curve["FLUX"]
ivar_obs = light_curve["FLUX_ERR"]**(-2)

#model.marginalized_log_likelihood(theta, t_obs, flux_obs, ivar_obs)


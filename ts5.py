


"""
Simulate light curves for test star 5 of http://iopscience.iop.org/article/10.1088/0004-637X/710/2/1596/pdf
"""

import numpy as np
import dateutil
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import mod_lightcurves as ml
import cadence

np.random.seed(123)


object_name = "TS5"


nu_max = 541e-6 # [Hz]
error_nu_max = 1e-6
delta_nu = 33.73e-6 # [Hz]
error_delta_nu = 0.5e-6 # [Hz]




# Construct frequencies and their standard deviations.
noise_sigma = 2 # ppm
r_01 = 0.5
amp_sigma = 0.7
d_k_01 = np.random.uniform(0.4, 0.6)
l_max, k_max = (1, 16)

nu_0 = nu_max
bell_h = amp_sigma ** 2
bell_w = delta_nu * nu_0 * 4.0

n_comp = (l_max + 1) * (2 * k_max + 1)
nus, amp_vars = ml.comb_freq_var(k_max, l_max, nu_0, delta_nu, nu_max, bell_h,
    bell_w, r_01, d_k_01)


# Generate signals.
amps = np.random.randn(n_comp * 2) * np.sqrt(amp_vars)
omegas = 2.0 * np.pi * nus


def savefigs(figure, basename):
    figure.savefig("{}.png".format(basename), dpi=300)
    figure.savefig("{}.pdf".format(basename), dpi=300)

    print("Created {0}.png\nCreated {0}.pdf".format(basename))


# marginalized log likelihood function
def mn_log_like_full_comb_marg(theta, t, data):

    c_mat = np.diag(np.ones(t.size) * noise_sigma ** 2)
    c_mat_inv = np.linalg.inv(c_mat)
    c_mat_log_det = np.linalg.slogdet(2.0 * np.pi * c_mat)[1]

    # construct frequencies and their std devs
    nus, amp_vars = ml.comb_freq_var(k_max, l_max, *theta)

    omegas = 2.0 * np.pi * nus

    # calculate log-like
    b_mat = ml.des_mat(t, omegas)
    v_inv_mat, log_det = ml.update_inv_det_stable(c_mat_inv, \
                                                  c_mat_log_det, \
                                                  b_mat, \
                                                  amp_vars)
    log_like = np.dot(data.T, np.dot(v_inv_mat, data)) + log_det
    return -0.5 * log_like


true_values = [nu_0, delta_nu, nu_max, bell_h, bell_w, r_01]
parameter_names = ("nu_0", "delta_nu", "nu_max", "bell_h", "bell_w", "r_01")

scales = dict(nu_max=1e-6, nu_0=1e-6, delta_nu=1e-6)
latex_labels = {
    "nu_0": r"$\nu_0$ $[\mu{\rm Hz}]$",
    "delta_nu": r"$\Delta\nu$ $[\mu{\rm Hz}]$",
    "nu_max": r"$\nu_{\rm max}$ $[\mu{\rm Hz}]$",
    "bell_h": "Bell height",
    "bell_w": "Bell width",
    "r_01": r"$r_{01}$"
}



grid_bounds = [
    # parameter_name, figure_prefix, grid_start, grid_end, grid_number
    ("nu_max", "nu_max", 100e-6, 260e-6, 1000),
    ("delta_nu", "delta_nu", 8e-6, 22e-6, 1000),
    ("nu_max", "nu_max_zoom", nu_max - 5e-6, nu_max + 5e-6, 1000),
    ("bell_h", "bell_h", 0.5 * bell_h, 2 * bell_h, 50),
    ("bell_w", "bell_w", 0.5 * bell_w, 2 * bell_w, 50),
    ("r_01", "r_01", 0.5 * r_01, 2 * r_01, 50)
]

cadences = [
    ("Gaia", cadence.gaia("KIC12008916_gaia_forecast.csv")),
]

for cadence, t in cadences:

    figure_title = "{} with {} observation cadence ($N_{{obs}} = {}$)".format(
        object_name, cadence, len(t))

    # Predict light curve given the observed times.
    data = np.sum(
        [ml.lightcurve(t, amps[2*i], amps[2*i + 1], omegas[i]) for i in range(n_comp)],
        axis=0)

    # Add noise.
    data += np.random.normal(0, noise_sigma, data.size)

    # Plot the predicted light curve.
    t_simulated = np.linspace(t.min(), t.max(), 1000)
    y_simulated = np.sum(
        [ml.lightcurve(t_simulated, amps[2*i], amps[2*i + 1], omegas[i]) \
            for i in range(n_comp)], axis=0)

    day = 24 * 60 * 60
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t_simulated/day, y_simulated, c="#3498db", lw=2, zorder=-1)

    # MPL HACK AHEAD MAGIC
    ax.scatter(t/day, data, facecolor="k", alpha=0.25 if cadence == "TESS" else 1)
    ax.errorbar(t/day, data, yerr=noise_sigma, fmt=None, c="k", alpha=0.25 if cadence == "TESS" else 1)

    ax.set_xlabel(r"${\rm Time}$ ${\rm since}$ ${\rm first}$ ${\rm observation},$ $\Delta{}t$ $[{\rm days}]$")
    ax.set_ylabel(r"${\rm Magnitude}$ ${\rm change},$ $\Delta{}G$ $[10^{-3}$ ${\rm mag}]$")

    ax.set_title(figure_title)

    ylim = np.abs(ax.get_ylim()).max()
    ax.set_ylim(-ylim, +ylim)

    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(5))

    ax.axhline(0, c="#666666", linestyle=":", zorder=-1)

    fig.tight_layout()
    savefigs(fig, "{}_{}_light_curve".format(object_name, cadence))

    # Now zoom in to 1/5th of the data.
    ax.set_xlim(0, np.diff(ax.get_xlim())/5)
    savefigs(fig, "{}_{}_light_curve_zoom".format(object_name, cadence))


    # Plot log-likelihood around true value.
    for parameter_name, prefix, start, end, num_points in grid_bounds:

        x = np.linspace(start, end, num_points)

        # Add the true value.
        index = parameter_names.index(parameter_name)
        true_value = true_values[index]
        x = np.sort(np.hstack([x, true_value]))

        ln_likelihood = np.ones(x.size)
        for i, value in enumerate(x):

            theta = np.array(true_values).copy()
            theta[index] = value

            if parameter_name == "nu_max":
                # Also set nu_0 as nu_max
                theta[0] = value

            ln_likelihood[i] = mn_log_like_full_comb_marg(theta, t, data)

        # Plot it.
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(x / scales.get(parameter_name, 1), ln_likelihood, c="k")

        ax.axvline(
            true_value / scales.get(parameter_name, 1), 
            c="#3498db", lw=3, zorder=-10)

        ax.set_xlabel(latex_labels.get(parameter_name, parameter_name))
        ax.set_ylabel(r"${\rm Log}$-${\rm likelihood,}$ $\mathcal{L}$")

        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))

        ax.set_title(figure_title)
        fig.tight_layout()

        savefigs(fig, "{}_{}_{}".format(object_name, cadence, parameter_name))


# TODO: 
# [ ] Repeat all for another observing cadence near the poles
# [ ] Repeat for the same star using LOTS of gaia observations
# [ ] Assign the amplitudes of the frequencies so that photometric precision can be related

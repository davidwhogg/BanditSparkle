

"""
Simulate light curves for KIC 12008916

"""

import numpy as np
import dateutil
from astropy.table import Table

import mod_lightcurves as ml

np.random.seed(123)

object_name = "kic12008916"

# Davies et al. (2016) -- https://arxiv.org/pdf/1601.02802.pdf
nu_max = 160.9e-6 # [Hz]
error_nu_max = 0.5e-6 # [Hz]
delta_nu = 12.89e-6 # [Hz]
error_delta_nu = 3e-6 # [Hz]


# Get the forecasted observation times from Gaia for this star
gaia_obs_time_column = "ObservationTimeAtGaia[UTC]"
gaia_forecast = Table.read(
    "{}_gaia_forecast.csv".format(object_name), format="csv")

# Predict the magnitude changes at the observed (delta) times.
first = dateutil.parser.parse(gaia_forecast[gaia_obs_time_column][0])
t = np.array([(dateutil.parser.parse(v) - first).total_seconds() \
        for v in gaia_forecast[gaia_obs_time_column]])


# Questions/figures:

# [ ] Plot log-likelihood with nu_max around grid points etc
# [ ] Plot light curve and data points.
# [ ] Do for another star with a very different nu_max
# [ ] Do for the kic12008916 but for a star that gets observed a *lot*
# [ ] Assign the amplitudes of the frequencies so that photometric precision can be related


# Switch to TESS-like precision.
t_min = 0.0
t_max = 30.0 * 24.0 * 60.0 # 30 days in mins
n_t = int((t_max - t_min) / ml.t_sample)
#t = np.linspace(t_min, t_max, 2500)


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


# Predict lightcurve at the observed points.
generated_data = np.zeros(t.size, dtype=float)
for i in range(n_comp):
    generated_data += ml.lightcurve(t, amps[2 * i], amps[2 * i + 1], omegas[i])

# Add noise.
generated_data += np.random.normal(0, noise_sigma, generated_data.size)

data = generated_data

c_mat = np.diag(np.ones(t.size) * noise_sigma ** 2)
c_mat_inv = np.linalg.inv(c_mat)
c_mat_log_det = np.linalg.slogdet(2.0 * np.pi * c_mat)[1]



def mn_log_like_full_comb_marg(theta, t):

    # construct frequencies and their std devs
    nus, amp_vars = ml.comb_freq_var(ml.k_max, ml.l_max, *theta)

    omegas = 2.0 * np.pi * nus

    # calculate log-like
    b_mat = ml.des_mat(t, omegas)
    v_inv_mat, log_det = ml.update_inv_det_stable(c_mat_inv, \
                                                  c_mat_log_det, \
                                                  b_mat, \
                                                  amp_vars)
    log_like = np.dot(data.T, np.dot(v_inv_mat, data)) + log_det
    return -0.5 * log_like



# Evaluate the marginalized log-likelihood.
# nu_0, d_nu, nu_max, bell_h, bell_w, r_01
theta = [nu_0, delta_nu, nu_max, bell_h, bell_w, r_01]

# Vary nu_max and delta_nu.

grid_nu_max = np.linspace(100, 260, 100) * 1e-6 # [Hz]
# Include true value
grid_nu_max = np.sort(np.hstack([grid_nu_max, nu_max]))

ln_ps = []
thetas = []

for nu_max_ in grid_nu_max:

    print(nu_max_)
    theta = [nu_max_, delta_nu, nu_max_, bell_h, bell_w, r_01]

    thetas.append(theta)
    ln_ps.append(mn_log_like_full_comb_marg(theta, t))

thetas = np.array(thetas)
ln_ps = np.array(ln_ps)


fig, ax = plt.subplots()
ax.scatter(thetas[:, 0] * 1e6, ln_ps)
ax.axvline(nu_max * 1e6, c='b')


offset = 0.5


ln_ps2 = []
thetas2 = []
mini_grid_nu_max = np.linspace(nu_max - offset * 1e-6, nu_max + offset * 1e-6, 100)
for nu_max_ in mini_grid_nu_max:
    thetas2.append([nu_max_, delta_nu, nu_max_, bell_h, bell_w, r_01])
    ln_ps2.append(mn_log_like_full_comb_marg(thetas2[-1], t))

thetas2 = np.array(thetas2)
ln_ps2 = np.array(ln_ps2)





fig, ax = plt.subplots()
ax.scatter(thetas2[:, 0] * 1e6, ln_ps2)
ax.plot(thetas2[:, 0] * 1e6, ln_ps2)
ax.axvline(nu_max * 1e6, c='b')



ln_ps3 = []
thetas3 = []
mini_grid_delta_nu = np.linspace(delta_nu - 5 * error_delta_nu, delta_nu + 5 * error_delta_nu, 100)
mini_grid_nu_max = np.linspace(nu_max - offset * 1e-6, nu_max + offset * 1e-6, 100)
for nu_max_ in mini_grid_nu_max:
    for delta_nu_ in mini_grid_delta_nu:
        thetas3.append([nu_max_, delta_nu_, nu_max_, bell_h, bell_w, r_01])
        ln_ps3.append(mn_log_like_full_comb_marg(thetas3[-1], t))

thetas3 = np.array(thetas3)
ln_ps3 = np.array(ln_ps3)


fig, ax = plt.subplots()
ax.scatter(thetas3[:, 0] * 1e6, ln_ps3)
ax.plot(thetas3[:, 0] * 1e6, ln_ps3)
ax.axvline(nu_max * 1e6, c='b')




fig, axes = plt.subplots(1, 2)
for i, ax in enumerate(axes):

    ax.scatter(thetas3[:, i] * 1e6, ln_ps3)
    #ax.plot(thetas2[:, i] * 1e6, ln_ps2)
    #indices = np.argsort(thetas2[:, i])
    #ax.plot(thetas2[:, i][indices] * 1e6, ln_ps2[indices])

axes[0].axvline(nu_max * 1e6, c='b')
axes[1].axvline(delta_nu * 1e6, c='b')





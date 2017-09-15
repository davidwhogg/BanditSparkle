import numpy as np
import matplotlib.pyplot as mp
import mod_lightcurves as ml

# plot settings
lw = 1.5
mp.rc('font', family = 'serif')
mp.rcParams['text.latex.preamble'] = [r'\boldmath']
mp.rcParams['axes.linewidth'] = lw
mp.rcParams['lines.linewidth'] = lw

# time sampling
s2d = 24.0 * 3600.0
t_tot = 8.0 * 30.0 * s2d
dt = 30.0 * 60.0
n_samples = int(t_tot / dt + 1)
n_samples = int(t_tot / dt)
#t = np.linspace(0, t_tot, n_samples)
t = np.arange(0, n_samples) * dt

# signal definition: exponential decay with given amp and timescale
ac_par = [70000.0, 45000.0]
k_mat = ml.col_noise_cov_mat(t, ac_par[0], ac_par[1])
s_t = ml.col_noise(k_mat)

# calculate the PSD of this input signal. divide FTs by n_samples 
# to get correct units
s_f = np.fft.rfft(s_t) / n_samples
psd = s_f * s_f.conj()
f = np.fft.rfftfreq(n_samples, dt)

# also calculate the expected PSD. calculate the autocorrelation as a
# function of lag, then wrap so that the timestream starts with an 
# exponential decay and ends with the rise. then take the FT, taking 
# care to include the normalization factor of 1 / n_samples
# NB: there's some shenanigans here about odd vs even number of time
# steps
lag = t - dt * n_samples / 2
#print lag[n_samples / 2 - 2: n_samples / 2 + 2]
ac_exp = ml.laplace(lag, ac_par[0], ac_par[1])
psd_exp = np.fft.rfft(np.roll(ac_exp, n_samples / 2)) / n_samples

# and finally plot the analytic PSD. this is the continuous FT of an
# exponentially decaying function. as it's calculated from an integral
# we need to divide by dt; as it's an FT, we need to divide by 
# n_samples too
om = 2.0 * np.pi * f
psd_ana = 2.0 * ac_par[0] / ac_par[1] / \
		 (1.0 / ac_par[1] ** 2 + om ** 2) / dt / n_samples

# now calculate the autocorrelation from the inverse FT of the PSD.
# normalize by n_samples to get correct units
ac_from_psd = np.fft.irfft(psd) * n_samples

# plot!
fig, axes = mp.subplots(3, 1, figsize=(8, 15))
axes[0].plot(t / s2d, s_t, 'k')
axes[0].set_title('{:d} samples'.format(n_samples))
axes[0].set_xlabel(r'$t\,[{\rm d}]$')
axes[0].set_ylabel(r'${\rm flux}$')
axes[0].set_xlim(t[0] / s2d, t[-1] / s2d)
axes[1].loglog(f, psd, 'k', label='sampled')
axes[1].loglog(f, psd_exp, 'b', label='DFT(analytic AC)')
axes[1].loglog(f, psd_ana, 'r', label='analytic')
axes[1].legend(loc='lower left')
axes[2].plot(lag / s2d, np.roll(ac_from_psd, n_samples / 2), 'k', \
			 label='sampled')
axes[2].plot(lag / s2d, ac_exp, 'r--', label='analytic')
axes[2].legend(loc='upper left')
axes[1].set_xlabel(r'$\nu\,[{\rm Hz}]$')
axes[1].set_ylabel(r'${\rm PSD}\,[{\rm Hz}^{-1}]$')
axes[1].set_xlim(f[0], f[-1])
axes[2].set_xlabel(r'$\tau\,[{\rm d}]$')
axes[2].set_ylabel(r'${\rm autocorrelation}$')
axes[2].set_xlim(lag[0] / s2d, lag[-1] / s2d)
mp.savefig('fourier_norms.pdf', bbox_inches='tight')
mp.show()

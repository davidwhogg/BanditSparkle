import numpy as np
import numpy.random as npr
import matplotlib.pyplot as mp
import mod_lightcurves as ml

# pyMultiNest imports
#from __future__ import absolute_import, unicode_literals, print_function
import pymultinest as mn
import os

# plot settings
lw = 1.5
mp.rc('font', family = 'serif')
mp.rcParams['text.latex.preamble'] = [r'\boldmath']
mp.rcParams['axes.linewidth'] = lw
mp.rcParams['lines.linewidth'] = lw

# settings
t_min = 0.0
t_max = 30.0 * 24.0 * 60.0 # 30 days in mins
n_t = int((t_max - t_min) / ml.t_sample)
n_comp = 2

# determine accessible frequency range
nu_min = 1.0 / (t_max - t_min)
nu_nyq = 0.5 * (n_t - 1) / (t_max - t_min)

# generate signal parameters
amps = npr.normal(ml.amp_mu, ml.amp_sigma, n_comp * 2)
if ml.model == 'ind':
	taus = 10.0 ** (npr.uniform(ml.log_tau_min, ml.log_tau_max, \
								n_comp))
	omegas = 2.0 * np.pi / taus
elif ml.model == 'comb' or ml.model == 'comb_marg':
	omega_c = 10.0 ** (npr.uniform(ml.log_omega_min, \
								   ml.log_omega_max))
	d_omega = npr.normal(ml.d_omega_mu, ml.d_omega_sigma)
	omegas = omega_c * \
			 (1.0 + d_omega * (np.arange(n_comp) - \
			 				   (n_comp - 1) / 2.0))

# generate data
t = np.linspace(t_min, t_max, n_t)
d = np.zeros(n_t)
for i in range(n_comp):
	d += ml.lightcurve(t, amps[2 * i], amps[2 * i + 1], omegas[i])
d += npr.normal(0.0, ml.noise_sigma, n_t)

# cheeky FFT
ft = np.fft.rfft(d)
om = 2.0 * np.pi * np.fft.rfftfreq(n_t, ml.t_sample)

# save data and ground truth parameters to file
print '== components =='
stdout_fmt = '{:8.6f} * sin({:8.6f} * t) + ' + \
			 '{:8.6f} * cos({:8.6f} * t)'
for i in range(n_comp):
	print stdout_fmt.format(amps[2 * i], omegas[i], \
							amps[2 * i + 1], omegas[i])
if ml.model == 'ind':
	file_fmt = '{:19.12e} {:19.12e} {:19.12e}\n'
elif ml.model == 'comb' or ml.model == 'comb_marg':
	file_fmt = '{:19.12e} {:19.12e}\n'
np.savetxt('test_lightcurve.txt', zip(t, d))
with open('test_params.txt', 'w') as f:
	if ml.model == 'ind':
		for i in range(n_comp):
			f.write(file_fmt.format(amps[2 * i], amps[2 * i + 1], \
									omegas[i]))
	elif ml.model == 'comb' or ml.model == 'comb_marg':
		f.write(file_fmt.format(omega_c, d_omega))
		for i in range(n_comp):
			f.write(file_fmt.format(amps[2 * i], amps[2 * i + 1]))
f.closed

# plot data and FFT
mp.rcParams["figure.figsize"] = [8, 10]
fig, axes = mp.subplots(3)
axes[0].step(t, d)
axes[0].set_xlim(t_min, t_max)
axes[0].set_xlabel(r'$t [{\rm s}]$')
axes[0].set_ylabel(r'$F [{\rm mag}]$')
axes[1].semilogx(om, np.absolute(ft))
for i in range(n_comp):
	axes[1].axvline(omegas[i], color = 'red', ls = '--')
axes[1].set_xlim(np.min(om), np.max(om))
axes[1].set_xlabel(r'$\omega [{\rm Hz}]$')
axes[1].set_ylabel(r'$|F| [{\rm mag}]$')
axes[2].semilogx(om, np.angle(ft))
for i in range(n_comp):
	axes[2].axvline(omegas[i], color = 'red', ls = '--')
axes[2].set_xlim(np.min(om), np.max(om))
axes[2].set_xlabel(r'$\nu [{\rm Hz}]$')
axes[2].set_ylabel(r'$\phi [{\rm rad}]$')
mp.savefig('test_input_plots.pdf', box_inches = 'tight')

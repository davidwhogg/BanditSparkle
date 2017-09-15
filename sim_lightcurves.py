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
t_max = 30.0 * 24.0 * 3600.0 # 30 days in s
n_t = int((t_max - t_min) / ml.t_sample)
constrain = True

# determine accessible frequency range
nu_min = 1.0 / (t_max - t_min)
nu_nyq = 0.5 * (n_t - 1) / (t_max - t_min)

# constrain the particular realisation if desired
if constrain:
	with open('test_rng_state.txt', 'r') as f:
		lines = f.readlines()
	f.closed
	state = (lines[0].rstrip(), \
			 np.array(lines[1:-3], dtype = 'uint32'), \
			 int(lines[-3]), int(lines[-2]), float(lines[-1]))
	np.random.set_state(state)
else:
	state = np.random.get_state()
	with open('test_rng_state.txt', 'w') as f:
		f.write(state[0] + '\n')
		for i in range(len(state[1])):
			f.write('{:10d}\n'.format(state[1][i]))
		f.write('{:10d}\n'.format(state[-3]))
		f.write('{:10d}\n'.format(state[-2]))
		f.write('{:3.1f}\n'.format(state[-1]))
	f.closed

# generate signal parameters
if ml.model == 'star':

	# number of modes
	n_comp = (ml.l_max + 1) * (2 * ml.k_max + 1)

	# draw parameters
	nu_0 = 10.0 ** (npr.uniform(ml.log_omega_min, \
								ml.log_omega_max)) / 2.0 / np.pi
	d_nu = npr.normal(ml.d_omega_mu, ml.d_omega_sigma) / 5.0
	nu_max = nu_0
	bell_h = ml.amp_sigma ** 2
	if ml.coloured_noise:
		bell_h *= 8000#2000
	bell_w = d_nu * nu_0 * 4.0
	r_01 = 0.5
	d_k_01 = npr.uniform(0.4, 0.6)

	# construct frequencies and their std devs
	nus, amp_vars = ml.comb_freq_var(ml.k_max, ml.l_max, nu_0, \
									 d_nu, nu_max, bell_h, \
									 bell_w, r_01, d_k_01)
	'''for i in range(n_comp):
		mp.plot([nus[i], nus[i]], [0, amp_sigmas[i] ** 2], 'k')
	nu_grid = np.linspace(nus[0], nus[-1], 1000)
	mp.plot(nu_grid, bell_h * np.exp(-0.5 * ((nu_grid - nu_max) / bell_w) ** 2), 'r')
	mp.plot(nu_grid, bell_h * np.exp(-0.5 * ((nu_grid - nu_max) / bell_w) ** 2) * r_01, 'r--')
	mp.show()
	exit()'''

	# generate signal
	amps = npr.randn(n_comp * 2) * np.sqrt(amp_vars)
	omegas = 2.0 * np.pi * nus

else:

	n_comp = 2
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
if ml.coloured_noise:
	k_mat = ml.col_noise_cov_mat(t, 67500.0, 45500.0)
	d += ml.col_noise(k_mat)
	d += npr.normal(0.0, 18.3, n_t)
else:
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
else:
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
	elif ml.model == 'star':
		par_fmt = '{:19.12e} {:19.12e} {:19.12e} {:19.12e} ' + \
				  '{:19.12e} {:19.12e} {:19.12e}\n'
		f.write(par_fmt.format(nu_0, d_nu, nu_max, bell_h, \
							   bell_w, r_01, d_k_01))
		for i in range(n_comp):
			f.write(file_fmt.format(amps[2 * i], amps[2 * i + 1]))
f.closed

# plot data and FFT
mp.rcParams["figure.figsize"] = [8, 10]
fig, axes = mp.subplots(3)
axes[0].step(t, d)
axes[0].set_xlim(t_min, t_max)
axes[0].set_xlabel(r'$t [{\rm s}]$')
axes[0].set_ylabel(r'$F [{\rm flux}]$')
if ml.coloured_noise:
	axes[1].loglog(om * 1e6, np.absolute(ft) ** 2)
else:
	axes[1].semilogx(om * 1e6, np.absolute(ft) ** 2)
for i in range(n_comp):
	axes[1].axvline(omegas[i] * 1e6, color = 'red', ls = '--')
axes[1].set_xlim(np.min(om) * 1e6, np.max(om) * 1e6)
axes[1].set_xlabel(r'$\omega [{\rm \mu Hz}]$')
axes[1].set_ylabel(r'$|F|^2 [{\rm flux}^2]$')
axes[2].semilogx(om * 1e6, np.angle(ft))
for i in range(n_comp):
	axes[2].axvline(omegas[i] * 1e6, color = 'red', ls = '--')
axes[2].set_xlim(np.min(om) * 1e6, np.max(om) * 1e6)
axes[2].set_xlabel(r'$\omega [{\rm \mu Hz}]$')
axes[2].set_ylabel(r'$\phi [{\rm rad}]$')
mp.savefig('test_input_plots.pdf', bbox_inches='tight')

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as mp
import mod_lightcurves as ml
import scipy.optimize as so

# pyMultiNest imports
#from __future__ import absolute_import, unicode_literals, print_function
import pymultinest as mn
import os

# MultiNest prior
def mn_prior(cube, n_dim, n_par):

	for i in range(n_comp_fit):
		cube[stride*i] = ml.gaussian_prior(cube[stride*i], \
										   ml.amp_mu, ml.amp_sigma)
		cube[stride*i+1] = ml.gaussian_prior(cube[stride*i+1], \
											 ml.amp_mu, ml.amp_sigma)
		tau = ml.log_uniform_prior(cube[stride*i+2], \
								   ml.log_tau_min, \
								   ml.log_tau_max)
		cube[stride*i+2] = 2.0 * np.pi / tau

# MultiNest comb prior
def mn_prior_comb(cube, n_dim, n_par):

	cube[0] = ml.log_uniform_prior(cube[0], ml.log_omega_min, \
								   ml.log_omega_max)
	cube[1] = ml.gaussian_prior(cube[1], ml.d_omega_mu, \
								ml.d_omega_sigma)
	for i in range(n_comp_fit):
		cube[2+stride*i] = ml.gaussian_prior(cube[2+stride*i], \
											 ml.amp_mu, ml.amp_sigma)
		cube[3+stride*i] = ml.gaussian_prior(cube[3+stride*i], \
											 ml.amp_mu, ml.amp_sigma)

# MultiNest comb prior, marginalized over amps
def mn_prior_comb_marg(cube, n_dim, n_par):

	cube[0] = ml.log_uniform_prior(cube[0], ml.log_omega_min, \
								   ml.log_omega_max)
	cube[1] = ml.gaussian_prior(cube[1], ml.d_omega_mu, \
								ml.d_omega_sigma)

# MultiNest full comb prior, marginalized over amps
def mn_prior_full_comb_marg(cube, n_dim, n_par):

	'''
	# initial priors: some (marked) are bad!
	# nu_0
	cube[0] = ml.log_uniform_prior(cube[0], ml.log_omega_min, \
								   ml.log_omega_max) / 2.0 / np.pi
	# d_nu 
	cube[1] = ml.gaussian_prior(cube[1], ml.d_omega_mu / 5.0, \
								ml.d_omega_sigma / 5.0)
	# nu_max
	cube[2] = ml.log_uniform_prior(cube[2], ml.log_omega_min, \
								   ml.log_omega_max) / 2.0 / np.pi
	# bell_h
	#cube[3] = ml.log_uniform_prior(cube[3], -2.0, 1.0)
	cube[3] = ml.log_uniform_prior(cube[3], 1.0, 3.0)
	# bell_w 
	cube[4] = ml.log_uniform_prior(cube[4], ml.log_omega_min, \
								   ml.log_omega_max) / 2.0 / np.pi
	# r_01
	cube[5] = ml.gaussian_prior(cube[5], 0.5, 0.1)
	# kappa_01
	cube[6] = ml.uniform_prior(cube[6], 0.2, 0.8)
	'''

	# nu_0
	cube[0] = ml.uniform_prior(cube[0], 10.0e-6, 200.0e-6)
	# d_nu
	cube[1] = ml.uniform_prior(cube[1], 0.05, 0.2)
	# nu_max
	cube[2] = ml.uniform_prior(cube[2], 10.0e-6, 200.0e-6)
	# bell_h
	cube[3] = ml.log_uniform_prior(cube[3], 0.0, 3.0)
	# bell_w
	cube[4] = ml.uniform_prior(cube[4], 1.0e-6, 50.0e-6)
	# r_01
	cube[5] = ml.gaussian_prior(cube[5], 0.5, 0.1)
	# kappa_01
	cube[6] = ml.uniform_prior(cube[6], 0.2, 0.8)

# MultiNest full comb prior, marginalized over amps, with nu_0 and 
# nu_max defined to be equal
def mn_prior_full_comb_marg_lock_nu(cube, n_dim, n_par):

	'''
	# initial priors: some (marked) are bad!
	# nu_0 = nu_max - range bad, not sure log is good
	cube[0] = ml.log_uniform_prior(cube[0], ml.log_omega_min, \
								   ml.log_omega_max) / 2.0 / np.pi
	# d_nu - just plain bad
	cube[1] = ml.gaussian_prior(cube[1], ml.d_omega_mu / 5.0, \
								ml.d_omega_sigma / 5.0)
	# bell_h
	cube[2] = ml.log_uniform_prior(cube[2], 1.0, 3.0)
	# bell_w
	cube[3] = ml.log_uniform_prior(cube[3], ml.log_omega_min, \
								   ml.log_omega_max) / 2.0 / np.pi
	# r_01
	cube[4] = ml.gaussian_prior(cube[4], 0.5, 0.1)
	# kappa_01
	cube[5] = ml.uniform_prior(cube[5], 0.2, 0.8)
	'''

	# nu_0 = nu_max
	cube[0] = ml.uniform_prior(cube[0], 10.0e-6, 200.0e-6)
	# d_nu
	cube[1] = ml.uniform_prior(cube[1], 0.05, 0.2)
	# bell_h
	cube[2] = ml.log_uniform_prior(cube[2], 0.0, 3.0)
	# bell_w
	cube[3] = ml.uniform_prior(cube[3], 1.0e-6, 50.0e-6)
	# r_01
	cube[4] = ml.gaussian_prior(cube[4], 0.5, 0.1)
	# kappa_01
	cube[5] = ml.uniform_prior(cube[5], 0.2, 0.8)

# MultiNest full comb prior taking central frequency to be nearest
# to nu_max, marginalized over amps
def mn_prior_full_comb_marg_nearest(cube, n_dim, n_par):

	# d_nu_0: dimensionless shift in frequency between nu_0 and nu_max
	cube[0] = ml.uniform_prior(cube[0], -0.5, 0.5)
	# d_nu: dimensionfull frequency spacing
	# maybe need log spacing here? investigate!
	cube[1] = ml.uniform_prior(cube[1], 1.0e-6, 40.0e-6)
	# nu_max
	cube[2] = ml.uniform_prior(cube[2], 10.0e-6, 200.0e-6)
	# bell_h
	cube[3] = ml.log_uniform_prior(cube[3], 0.0, 3.0)
	# bell_w
	cube[4] = ml.uniform_prior(cube[4], 1.0e-6, 50.0e-6)
	# r_01
	cube[5] = ml.gaussian_prior(cube[5], 0.5, 0.1)
	# kappa_01
	cube[6] = ml.uniform_prior(cube[6], 0.2, 0.8)

# MultiNest full comb prior taking central frequency to be nearest
# to nu_max, marginalized over amps. l_max is general
def mn_prior_full_comb_marg_nearest_gen(cube, n_dim, n_par):

	# d_nu_0: dimensionless shift in frequency between nu_0 and nu_max
	cube[0] = ml.uniform_prior(cube[0], -0.5, 0.5)
	# d_nu: dimensionfull frequency spacing
	# maybe need log spacing here? investigate!
	cube[1] = ml.uniform_prior(cube[1], 1.0e-6, 40.0e-6)
	# nu_max
	cube[2] = ml.uniform_prior(cube[2], 10.0e-6, 200.0e-6)
	# bell_h
	cube[3] = ml.log_uniform_prior(cube[3], 0.0, 3.0)
	# bell_w
	cube[4] = ml.uniform_prior(cube[4], 1.0e-6, 50.0e-6)
	# r_0l
	for el in range(1, ml.l_max):
		ind = 4 + el
		#cube[ind] = ml.gaussian_prior(cube[ind], 0.5, 0.1)
		cube[ind] = ml.uniform_prior(cube[ind], 0.05, 1.0)
	# kappa_0l
	for el in range(1, ml.l_max):
		ind = 4 + ml.l_max + el
		cube[ind] = ml.uniform_prior(cube[ind], 0.05, 0.95)

# MultiNest log-likelihood
def mn_log_like(cube, n_dim, n_par):

	m = np.zeros(n_t)
	for i in range(n_comp_fit):
		m += ml.lightcurve(t, cube[stride*i], cube[stride*i+1], \
						   cube[stride*i+2])
	log_like = (-0.5 * ((d - m) / ml.noise_sigma) ** 2).sum()
	return log_like

# MultiNest comb log-likelihood
def mn_log_like_comb(cube, n_dim, n_par):

	m = np.zeros(n_t)
	omegas = cube[0] * \
			 (1.0 + cube[1] * (np.arange(n_comp_fit) - \
			 				   (n_comp_fit - 1) / 2.0))
	for i in range(n_comp_fit):
		m += ml.lightcurve(t, cube[2+stride*i], cube[3+stride*i], \
						   omegas[i])
	log_like = (-0.5 * ((d - m) / ml.noise_sigma) ** 2).sum()
	return log_like

# MultiNest comb log-likelihood, marginalized over amps
def mn_log_like_comb_marg(cube, n_dim, n_par):

	omegas = cube[0] * \
			 (1.0 + cube[1] * (np.arange(n_comp_fit) - \
			 				   (n_comp_fit - 1) / 2.0))
	b_mat = ml.des_mat(t, omegas)
	v_inv_mat, log_det = ml.update_inv_det(c_mat_inv, \
										   c_mat_log_det, b_mat, \
										   np.ones(n_comp_fit * 2) * \
										   ml.amp_sigma ** 2)
	log_like = np.dot(d.T, np.dot(v_inv_mat, d)) + log_det
	return -0.5 * log_like

# MultiNest comb log-likelihood, marginalized over amps
def mn_log_like_full_comb_marg(cube, n_dim, n_par):

	# construct frequencies and their std devs
	nus, amp_vars = ml.comb_freq_var(ml.k_max, ml.l_max, cube[0], \
									 cube[1], cube[2], cube[3], \
									 cube[4], cube[5], cube[6])
	omegas = 2.0 * np.pi * nus

	# calculate log-like
	b_mat = ml.des_mat(t, omegas)
	v_inv_mat, log_det = ml.update_inv_det_stable(c_mat_inv, \
												  c_mat_log_det, \
												  b_mat, \
												  amp_vars)
	log_like = np.dot(d.T, np.dot(v_inv_mat, d)) + log_det
	if ml.opt:
		return log_like
	else:
		return -0.5 * log_like

# MultiNest comb log-likelihood, marginalized over amps, with nu_0 
# and nu_max defined to be equal
def mn_log_like_full_comb_marg_lock_nu(cube, n_dim, n_par):

	# construct frequencies and their std devs
	nus, amp_vars = ml.comb_freq_var(ml.k_max, ml.l_max, cube[0], \
									 cube[1], cube[0], cube[2], \
									 cube[3], cube[4], cube[5])
	omegas = 2.0 * np.pi * nus

	# calculate log-like
	b_mat = ml.des_mat(t, omegas)
	v_inv_mat, log_det = ml.update_inv_det_stable(c_mat_inv, \
												  c_mat_log_det, \
												  b_mat, \
												  amp_vars)
	log_like = np.dot(d.T, np.dot(v_inv_mat, d)) + log_det
	if ml.opt:
		return log_like
	else:
		return -0.5 * log_like

# MultiNest full comb log-likelihood taking central frequency to be 
# nearest to nu_max, marginalized over amps
def mn_log_like_full_comb_marg_nearest(cube, n_dim, n_par):

	# construct frequencies and their std devs
	nus, amp_vars = ml.comb_freq_var_nearest(ml.k_max, ml.l_max, \
											 cube[0], cube[1], \
											 cube[2], cube[3], \
											 cube[4], cube[5], \
											 cube[6])
	omegas = 2.0 * np.pi * nus

	# calculate log-like
	b_mat = ml.des_mat(t, omegas)
	v_inv_mat, log_det = ml.update_inv_det_stable(c_mat_inv, \
												  c_mat_log_det, \
												  b_mat, \
												  amp_vars)
	log_like = np.dot(d.T, np.dot(v_inv_mat, d)) + log_det
	if ml.opt:
		return log_like
	else:
		return -0.5 * log_like

# MultiNest full comb log-likelihood taking central frequency to be 
# nearest to nu_max, marginalized over amps. allow for an arbitrary 
# number of ells
def mn_log_like_full_comb_marg_nearest_gen(cube, n_dim, n_par):

	# construct frequencies and their std devs
	r_0l = cube[5: 5 + ml.l_max]
	d_k_0l = cube[5 + ml.l_max:]
	nus, amp_vars = ml.comb_freq_var_nearest_gen(ml.k_max, \
												 ml.l_max, \
												 cube[0], cube[1], \
												 cube[2], cube[3], \
												 cube[4], r_0l, \
												 d_k_0l)
	omegas = 2.0 * np.pi * nus

	# calculate log-like
	b_mat = ml.des_mat(t, omegas)
	v_inv_mat, log_det = ml.update_inv_det_stable(c_mat_inv, \
												  c_mat_log_det, \
												  b_mat, \
												  amp_vars)
	log_like = np.dot(d.T, np.dot(v_inv_mat, d)) + log_det
	if ml.opt:
		return log_like
	else:
		return -0.5 * log_like

# sims or data?
cat_id = 7674224 # 6757558 # None #1162746 # or None

# read data
if cat_id is not None:
	td = np.genfromtxt('test_{:d}_lightcurve.txt'.format(cat_id))
else:
	td = np.genfromtxt('test_lightcurve.txt')
n_t = td.shape[0]
t = td[:, 0]
d = td[:, 1]
if cat_id is not None:
	e = td[:, 2]

# create chains directory if needed
if not os.path.exists("chains"): os.mkdir("chains")

# go for it!
# NB: function call is 
#     ml.run(LogLikelihood, Prior, n_dims, n_params=None, 
#            n_clustering_params=None, wrapped_params=None, 
#            importance_nested_sampling=True, multimodal=True, 
#            const_efficiency_mode=False, n_live_points=1000, 
#            evidence_tolerance=0.5, sampling_efficiency=0.8, 
#            n_iter_before_update=100, null_log_evidence=-1e+90, 
#            max_modes=100, mode_tolerance=-1e+90, 
#            outputfiles_basename=u'chains/1-', seed=-1, 
#            verbose=False, resume=True, context=0, 
#            write_output=True, log_zero=-1e+100, max_iter=0, 
#            init_MPI=False, dump_callback=None)
n_comp_fit = 2
if ml.model == 'ind':
	n_par_fit = n_comp_fit * ml.n_p
	stride = ml.n_p
	mn.run(mn_log_like, mn_prior, n_par_fit, resume = False, \
		   verbose = True, outputfiles_basename = u'chains/test', \
		   n_live_points = 1000, evidence_tolerance = 0.5, \
		   sampling_efficiency = 0.3)
elif ml.model == 'comb':
	n_par_fit = 2 + n_comp_fit * (ml.n_p - 1)
	stride = ml.n_p - 1
	mn.run(mn_log_like_comb, mn_prior_comb, n_par_fit, \
		   resume = False, verbose = True, \
		   outputfiles_basename = u'chains/test', \
		   n_live_points = 1000, evidence_tolerance = 0.5, \
		   sampling_efficiency = 0.3)
elif ml.model == 'comb_marg':
	n_par_fit = 2
	c_mat = np.diag(np.ones(n_t) * ml.noise_sigma ** 2)
	c_mat_inv = np.linalg.inv(c_mat)
	c_mat_log_det = np.linalg.slogdet(2.0 * np.pi * c_mat)[1]
	mn.run(mn_log_like_comb_marg, mn_prior_comb_marg, n_par_fit, \
		   resume = False, verbose = True, \
		   outputfiles_basename = u'chains/test', \
		   n_live_points = 1000, evidence_tolerance = 0.5, \
		   sampling_efficiency = 0.3)
elif ml.model == 'star':
	if ml.nu_max_eq_nu_0:
		n_par_fit = 4 + 2 * ml.l_max
	else:
		n_par_fit = 5 + 2 * ml.l_max
	if cat_id is not None:
		#c_mat = np.diag(e ** 2 * 722.597) # coloured noise boost
		p_opt = np.genfromtxt('test_{:d}_autocorr_fit.txt'.format(cat_id))
		k_mat = ml.col_noise_cov_mat(t, *p_opt)
		c_mat = np.diag(e ** 2) + k_mat
	elif ml.coloured_noise:
		k_mat = ml.col_noise_cov_mat(t, 67500.0, 45500.0)
		c_mat = np.diag(np.ones(n_t) * 18.3 ** 2) + k_mat
	else:
		c_mat = np.diag(np.ones(n_t) * ml.noise_sigma ** 2)
	c_mat_inv = np.linalg.inv(c_mat)
	c_mat_log_det = np.linalg.slogdet(2.0 * np.pi * c_mat)[1]
	if ml.opt:
		print 'optimizing!'
		if ml.nu_max_eq_nu_0:
			# nu_0=nu_max, d_nu, bell_h, bell_w, r_01, kappa_01
			x0 = (27.98e-6, 0.14, 100.0, 4.95e-6, 0.5, 0.5)
			results = so.minimize(mn_log_like_full_comb_marg_lock_nu,\
								  x0, args = (n_par_fit, n_par_fit), \
								  method = 'Nelder-Mead', \
								  options = {'maxfev': 10000})
		else:
			# nu_0, d_nu, nu_max, bell_h, bell_w, r_01, kappa_01
			#x0 = (2.071229237821e-03, 5.977924454460e-02, \
			#	  2.071229237821e-03, 4.900000000000e-01, \
			#	  4.952660764624e-04, 5.000000000000e-01, \
			#	  3.333333333333e-01)
			x0 = (26.30e-6, 0.14, 27.98e-6, 500.0, \
				  4.95e-06, 0.5, 0.5)
			results = so.minimize(mn_log_like_full_comb_marg, x0, \
								  args = (n_par_fit, n_par_fit), \
								  method = 'Nelder-Mead', \
								  options = {'maxfev': 10000})
		if results.success:
			print 'converged ({:d} func evals)'.format(results.nfev)
			print 'minimum:'
			print results.x
			print 'loglike: {:19.12e}'.format(-0.5 * results.fun)
		else:
			print results
	else:
		print 'sampling!'
		if ml.nu_max_eq_nu_0:
			mn.run(mn_log_like_full_comb_marg_lock_nu, \
				   mn_prior_full_comb_marg_lock_nu, \
				   n_par_fit, resume = False, verbose = True, \
				   outputfiles_basename = u'chains/test_lock_nu_{:d}'.format(cat_id), \
				   n_live_points = 1000, evidence_tolerance = 0.5, \
				   sampling_efficiency = 0.3)
		else:
			'''mn.run(mn_log_like_full_comb_marg, mn_prior_full_comb_marg, \
				   n_par_fit, resume = False, verbose = True, \
				   outputfiles_basename = u'chains/test_{:d}'.format(cat_id), \
				   n_live_points = 1000, evidence_tolerance = 0.5, \
				   sampling_efficiency = 0.3)'''
			'''mn.run(mn_log_like_full_comb_marg_nearest, \
				   mn_prior_full_comb_marg_nearest, n_par_fit, \
				   resume = False, verbose = True, \
				   outputfiles_basename = u'chains/test_{:d}'.format(cat_id), \
				   n_live_points = 1000, evidence_tolerance = 0.5, \
				   sampling_efficiency = 0.3)'''
			mn.run(mn_log_like_full_comb_marg_nearest_gen, \
				   mn_prior_full_comb_marg_nearest_gen, n_par_fit, \
				   resume = False, verbose = True, \
				   outputfiles_basename = u'chains/test_{:d}'.format(cat_id), \
				   n_live_points = 1000, evidence_tolerance = 0.5, \
				   sampling_efficiency = 0.3)

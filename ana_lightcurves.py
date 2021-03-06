import numpy as np
import matplotlib.pyplot as mp
import mod_lightcurves as ml

# settings
cat_id = 1162746
cat_id = 6757558
#cat_id = 6104786
cat_id = 7674224
use_corner = True

# read in chains. column structure is 
# sample probability, -2*log(like), params
base = 'test'
if ml.nu_max_eq_nu_0:
	base += '_lock_nu'
if ml.model == 'star':
	if cat_id is not None:
		base += '_{:d}'.format(cat_id)
s = np.genfromtxt('chains/1440_samples_lock_nu/' + base + '.txt')[:, 2:]
n_s, n_p = s.shape
if ml.model == 'ind':
	stride = ml.n_p
	n_comp_fit = n_p / stride
elif ml.model == 'comb':
	stride = ml.n_p - 1
	n_comp_fit = (n_p - 2) / stride

# generate parameter labels
labels = []
if ml.model == 'star':
	if ml.nu_max_eq_nu_0:
		labels = [r'$\nu_{n00}$', r'$\Delta$', r'$H$', r'$W$', \
				  r'$r_{01}$', r'$\kappa_{01}$']
	else:
		labels = [r'$\nu_{n00}$', r'$\Delta$', r'$\nu_{\rm max}$', \
				  r'$H$', r'$W$', r'$r_{01}$', r'$\kappa_{01}$']
elif ml.model == 'comb' or ml.model == 'comb_marg':
	labels = [r'$\omega_0$', r'$\Delta\omega$']
if ml.model == 'ind' or ml.model == 'comb':
	for i in range(n_comp_fit):
		labels.append(r'$A_{:d}$'.format(i+1))
		labels.append(r'$B_{:d}$'.format(i+1))
		if ml.model == 'ind':
			labels.append(r'$\log (\omega_{:d})$'.format(i+1))

# read in ground truth
if ml.model == 'star':
	par_file = base + '_params.txt'
	with open(par_file, 'r') as f:
		p_true = []
		for p in f.readline().split():
			p_true.append(float(p))
		p_true = np.array(p_true)
	f.closed
else:
	p_true = np.genfromtxt('test_params.txt')
	n_comp_true, n_p_true = p_true.shape
	if ml.model == 'comb':
		n_comp_true -= 1

# choose between analysis software
if use_corner:

	# import module and produce core corner plot
	import corner as co
	if ml.model == 'ind':
		for i in range(n_comp_fit):
			s[:, stride*i+2] = np.log10(s[:, stride*i+2])
	elif ml.model == 'star':
		if ml.nu_max_eq_nu_0:
			s[:, 0] *= 1.0e6
			#s[:, 2] = np.log10(s[:, 2])
			s[:, 3] *= 1.0e6
			p_true[0] *= 1.0e6
			#p_true[2] = np.log10(p_true[2])
			p_true[3] *= 1.0e6
			#labels[2] = r'$\log_{10}H$'
		else:
			s[:, 0] *= 1.0e6
			s[:, 2] *= 1.0e6
			s[:, 3] = np.log10(s[:, 3])
			s[:, 4] *= 1.0e6
			p_true[0] *= 1.0e6
			p_true[2] *= 1.0e6
			p_true[3] = np.log10(p_true[3])
			p_true[4] *= 1.0e6
			labels[3] = r'$\log_{10}H$'
	#elif ml.model == 'comb':
	#	s[:, 0] = np.log10(s[:, 0])
	fig = co.corner(s, bins = 50, labels = labels, \
					plot_density = False, plot_contours = False, \
					no_fill_contours = True)
	
	# overplot ground truth
	if ml.model == 'ind':
		for i in range(n_comp_true):
			p_true[i, 2] = np.log10(p_true[i, 2])
	for i in range(n_p):
		for j in range(i):
			a_ind = i * n_p + j
			if ml.model == 'ind':
				p_ind = np.mod(i, stride)
				for k in range(n_comp_true):
					fig.axes[a_ind].axhline(p_true[k, p_ind], \
											color='red', ls=':', \
											lw = 1.5)
			elif ml.model == 'comb':
				if i < 2:
					fig.axes[a_ind].axhline(p_true[0, i], \
											color='red', ls=':', \
											lw = 1.5)
				else:
					p_ind = np.mod(i - 2, stride)
					for k in range(n_comp_true):
						fig.axes[a_ind].axhline(p_true[k+1, p_ind], \
												color='red', ls=':', \
												lw = 1.5)
			elif ml.model == 'comb_marg':
				fig.axes[a_ind].axhline(p_true[0, i], \
										color='red', ls=':', \
										lw = 1.5)
			elif ml.model == 'star':
				fig.axes[a_ind].axhline(p_true[i], \
										color='red', ls=':', \
										lw = 1.5)
		for j in range(i + 1):
			a_ind = i * n_p + j
			if ml.model == 'ind':
				p_ind = np.mod(j, stride)
				for k in range(n_comp_true):
					fig.axes[a_ind].axvline(p_true[k, p_ind], \
											color='red', ls=':', \
											lw = 1.5)
			elif ml.model == 'comb':
				if j < 2:
					fig.axes[a_ind].axvline(p_true[0, j], \
											color='red', ls=':', \
											lw = 1.5)
				else:
					p_ind = np.mod(j - 2, stride)
					for k in range(n_comp_true):
						fig.axes[a_ind].axvline(p_true[k+1, p_ind], \
												color='red', ls=':', \
												lw = 1.5)
			elif ml.model == 'comb_marg':
				fig.axes[a_ind].axvline(p_true[0, j], \
										color='red', ls=':', \
										lw = 1.5)
			elif ml.model == 'star':
				fig.axes[a_ind].axvline(p_true[j], \
										color='red', ls=':', \
										lw = 1.5)
	mp.savefig(base + '_posterior.pdf')

	# produce corner plot for parameters of interest
	if ml.model == 'star':

		# \Delta_\nu is a derived parameter
		if ml.nu_max_eq_nu_0:
			i_nu_0, i_delta, i_nu_max = (0, 1, 0)
		else:
			i_nu_0, i_delta, i_nu_max = (0, 1, 2)
		s_marg = np.zeros((n_s, 2))
		s_marg[:, 0] = s[:, i_nu_0] * s[:, i_delta]
		s_marg[:, 1] = s[:, i_nu_max]
		p_true_marg = [ p_true[i_nu_0] * p_true[i_delta], \
						p_true[i_nu_max] ]
		n_p_marg = len(p_true_marg)
		labels_marg = [ r'$\Delta_\nu$', r'$\nu_{\rm max}$' ]
		fig = co.corner(s_marg, bins = 50, labels = labels_marg, \
						plot_density = False, plot_contours = False, \
						no_fill_contours = True)
		for i in range(n_p_marg):
			for j in range(i):
				a_ind = i * n_p_marg + j
				fig.axes[a_ind].axhline(p_true_marg[i], \
										color='red', ls=':', \
										lw = 1.5)
			for j in range(i + 1):
				a_ind = i * n_p_marg + j
				fig.axes[a_ind].axvline(p_true_marg[j], \
										color='red', ls=':', \
										lw = 1.5)
		mp.savefig(base + '_posterior_marg.pdf')


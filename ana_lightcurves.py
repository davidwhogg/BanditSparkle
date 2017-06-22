import numpy as np
import matplotlib.pyplot as mp
import mod_lightcurves as ml

# settings
use_corner = True

# read in chains. column structure is 
# sample probability, -2*log(like), params
s = np.genfromtxt('chains/test.txt')[:, 2:]
n_s, n_p = s.shape
if ml.model == 'ind':
	stride = ml.n_p
	n_comp_fit = n_p / stride
elif ml.model == 'comb':
	stride = ml.n_p - 1
	n_comp_fit = (n_p - 2) / stride

# generate parameter labels
labels = []
if ml.model == 'comb' or ml.model == 'comb_marg':
	labels = [r'$\omega_0$', r'$\Delta\omega$']
if ml.model != 'comb_marg':
	for i in range(n_comp_fit):
		labels.append(r'$A_{:d}$'.format(i+1))
		labels.append(r'$B_{:d}$'.format(i+1))
		if ml.model == 'ind':
			labels.append(r'$\log (\omega_{:d})$'.format(i+1))

# read in ground truth
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
	mp.savefig('test_posterior.pdf')


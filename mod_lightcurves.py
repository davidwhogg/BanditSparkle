import numpy as np
import scipy.special as ss

# settings
opt = False
t_sample = 30.0 * 60.0
t_int = 30.0 * 60.0 # integration time per sample
tau_min = 3.0 * 3600.0
tau_max = 300.0 * 3600.0
log_tau_min = np.log10(tau_min)
log_tau_max = np.log10(tau_max)
l_max = 1
k_max = 16
omega_min = 2.0 * np.pi / tau_max
omega_max = 2.0 * np.pi / tau_min
log_omega_min = np.log10(omega_min)
log_omega_max = np.log10(omega_max)
d_omega_mu = 0.3
d_omega_sigma = 0.05
amp_min = 0.4
amp_max = 0.6
amp_mu = 0.0
amp_sigma = 0.7
noise_sigma = 2.0
coloured_noise = True
n_p = 3
model = 'star' # 'star', comb_marg', comb' or 'ind'
nu_max_eq_nu_0 = False

# lightcurve model
def lightcurve(time, amp_s, amp_c, omega):

	return amp_s * np.sin(omega * time) + \
		   amp_c * np.cos(omega * time)

# design matrix
def des_mat(time, omega):

	n_obs = len(time)
	n_par = len(omega) * 2
	des_mat = np.zeros((n_obs, n_par))
	for i in range(n_par / 2):
		des_mat[:, 2*i] = np.sin(omega[i] * time)
		des_mat[:, 2*i+1] = np.cos(omega[i] * time)
	return des_mat

# generate comb of frequencies and their variances
def comb_freq_var(k_max, l_max, nu_0, d_nu, nu_max, bell_h, bell_w, \
				  r_01, d_k_01 = 0.5):

	n_comp = (l_max + 1) * (2 * k_max + 1)
	ks = np.zeros(n_comp)
	ind = 0
	for k in range(-k_max, k_max + 1):
		for el in range(l_max + 1):
			ks[ind] = k + el * d_k_01
			ind += 1
	els = np.mod(np.arange(n_comp), 2)
	nus = nu_0 * (1.0 + ks * d_nu)
	amp_vars = bell_h * r_01 ** els * \
			   np.exp(-0.5 * ((nus - nu_max) / bell_w) ** 2)
	return nus, np.repeat(amp_vars, 2)

# generate comb of frequencies and their variances. rather than 
# explicitly define the central frequency, here we identify it as the 
# frequency within 0.5 delta_nu of nu_max
def comb_freq_var_nearest(k_max, l_max, d_nu_0, d_nu, nu_max, \
						  bell_h, bell_w, r_01, d_k_01 = 0.5):

	n_comp = (l_max + 1) * (2 * k_max + 1)
	nus = np.zeros(n_comp)
	ind = 0
	for k in range(-k_max, k_max + 1):
		for el in range(l_max + 1):
			nus[ind] = nu_max + d_nu * (d_nu_0 + k + el * d_k_01)
			ind += 1
	els = np.mod(np.arange(n_comp), 2)
	amp_vars = bell_h * r_01 ** els * \
			   np.exp(-0.5 * ((nus - nu_max) / bell_w) ** 2)
	return nus, np.repeat(amp_vars, 2)

# generate comb of frequencies and their variances. rather than 
# explicitly define the central frequency, here we identify it as the 
# frequency within 0.5 delta_nu of nu_max. allow for an arbitrary 
# number of ells
def comb_freq_var_nearest_gen(k_max, l_max, d_nu_0, d_nu, nu_max, \
							  bell_h, bell_w, r_0l, d_k_0l):
	
	# add ell = 0 mode to relevant arrays
	r_0l = np.insert(r_0l, 0, 1.0)
	d_k_0l = np.insert(d_k_0l, 0, 0.0)

	# form comb
	n_comp = (l_max + 1) * (2 * k_max + 1)
	nus = np.zeros(n_comp)
	amp_vars = np.zeros(n_comp)
	ind = 0
	for k in range(-k_max, k_max + 1):
		for el in range(l_max + 1):
			nus[ind] = nu_max + d_nu * (d_nu_0 + k + d_k_0l[el])
			amp_vars[ind] = bell_h * r_0l[el] * \
							np.exp(-0.5 * ((nus[ind] - nu_max) / \
										   bell_w) ** 2)
			ind += 1

	return nus, np.repeat(amp_vars, 2)

# various generic prior rescalings for MultiNest
def uniform_prior(x, x_min, x_max):
	
	return x_min + x * (x_max - x_min)

def log_uniform_prior(x, log_x_min, log_x_max):
	
	return 10.0 ** (log_x_min + x * (log_x_max - log_x_min))

def gaussian_prior(x, mu, sigma):

	return mu + np.sqrt(2.0) * sigma * ss.erfinv(2.0 * x - 1.0)

# matrix updates
def update_inv_det(c_mat_inv, c_mat_log_det, b_mat, par_vars):

	# par_vars should be an array of parameter variances
	lambda_mat_log_det = np.sum(np.log(2.0 * np.pi * par_vars))
	lambda_mat_inv = np.diag(1.0 / par_vars)

	# intermediate step to generate determinant update
	a_mat = lambda_mat_inv + np.dot(b_mat.T, np.dot(c_mat_inv, b_mat))
	a_mat_log_det = np.linalg.slogdet(a_mat / 2.0 / np.pi)[1]
	out_log_det = a_mat_log_det + lambda_mat_log_det + c_mat_log_det

	# update inverse
	a_mat_inv = np.linalg.inv(a_mat)
	c_inv_b = np.dot(c_mat_inv, b_mat)
	out_inv = c_mat_inv - np.dot(c_inv_b, np.dot(a_mat_inv, c_inv_b.T))
	return out_inv, out_log_det

def update_inv_det_stable(c_mat_inv, c_mat_log_det, b_mat, par_vars):

	# @TODO: speed up multiplication by diagonal matrix
	# @TODO: speed up addition with diagonal matrix

	# par_vars should be an array of parameter variances
	lambda_mat = np.diag(par_vars)
	q_mat_inv = np.dot(b_mat.T, np.dot(c_mat_inv, b_mat))
	a_mat = np.eye(len(par_vars)) + np.dot(lambda_mat, q_mat_inv)

	# update inverse
	a_mat_inv_lambda = np.dot(np.linalg.inv(a_mat), lambda_mat)
	c_inv_b = np.dot(c_mat_inv, b_mat)
	out_inv = c_mat_inv - np.dot(c_inv_b, np.dot(a_mat_inv_lambda, c_inv_b.T))

	# update determinant
	out_log_det = np.linalg.slogdet(a_mat)[1] + c_mat_log_det
	return out_inv, out_log_det

def laplace(x, amp, timescale):
	return amp * np.exp(-np.abs(x) / timescale)

def autocorrelate(x):
	return np.correlate(x, x, mode='full') / len(x)

def col_noise_cov_mat(times, amp, timescale):
	n_samples = len(times)
	k_mat = np.zeros((n_samples, n_samples))
	for i in range(n_samples):
		k_mat[i, :] = laplace(times[i] - times, amp, timescale)
	return k_mat

def col_noise(cov_mat, decomposed=False):
	if not decomposed:
		cov_mat = np.linalg.cholesky(cov_mat)
	wn = np.random.randn(cov_mat.shape[0])
	return np.dot(cov_mat, wn)


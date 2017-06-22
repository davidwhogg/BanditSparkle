import numpy as np
import scipy.special as ss

# settings
t_sample = 30.0
t_int = 30.0 # integration time per sample
tau_min = 3.0 * 60.0 # mins
tau_max = 300.0 * 60.0
log_tau_min = np.log10(tau_min)
log_tau_max = np.log10(tau_max)
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
n_p = 3
model = 'comb_marg' # 'comb_marg', comb' or 'ind'

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


"""
Forward-model light curves for giant stars with Solar-like oscillators.
"""

import numpy as np


class AsteroseismicModel(object):

    def __init__(self):
        return None


    def __call__(self, *args, **kwargs):
        return self.predict_light_curve(self, *args, **kwargs)


    def predict_power_spectrum(self, t, nu_max, delta_nu, **kwargs):
        r"""
        Predict the power spectrum for a given star.
        """

        # TODO HACK ASSUMES EVEN OBSERVATIONS
        omega = 2 * np.pi * np.fft.rfftfreq(t.size, np.ptp(t)/t.size)
        ft = np.fft.rfft(self.predict_light_curve(t, nu_max, delta_nu, **kwargs))

        return (omega, ft)


    def comb_frequencies(self, nu_max, delta_nu, l_max=1, k_max=16, 
        r_01=0.5, amp_sigma=0.7, d_k_01=0.5, nu_0=None, bell_height=None, 
        bell_width=None, exposure_times=None, **kwargs):

        nu_0 = nu_0 or nu_max
        bell_height = bell_height or amp_sigma**2
        bell_width = bell_width or 4 * delta_nu * nu_0

        n_components = (l_max + 1) * (2 * k_max + 1)

        ks = np.zeros(n_components)
        ind = 0
        for k in range(-k_max, k_max + 1):
            for el in range(l_max + 1):
                ks[ind] = k + el * d_k_01
                ind += 1

        els = np.mod(np.arange(n_components), 2)
        nus = nu_0 * (1.0 + ks * delta_nu)
        amp_vars = bell_height * r_01 ** els * \
                   np.exp(-0.5 * ((nus - nu_max) / bell_width) ** 2)
        amp_vars = np.repeat(amp_vars, 2)

        return (nus, amp_vars)


    def predict_light_curve(self, t, nu_max, delta_nu, l_max=1, k_max=16, 
        r_01=0.5, amp_sigma=0.7, d_k_01=0.5, nu_0=None, bell_height=None, 
        bell_width=None, amplitudes=None, exposure_times=None, **kwargs):
        r"""
        Predict the light curve for the given observation times.

        :param t:

        :param nu_max:
            The frequency of maximum power [Hz].

        :param delta_nu:
            The large period separation [Hz].

        :param l_max: [optional]
            The maximum number of l-modes to model (default: 1).

        :param k_max: [optional]
            The maximum number of k-modes to model (default: 16).

        # TODO
        """

        # Default values for nu_0, bell_width, bell_height
        nus, amp_vars = self.comb_frequencies(nu_max, delta_nu, l_max=l_max,
            k_max=k_max, r_01=r_01, amp_sigma=amp_sigma, d_k_01=d_k_01,
            nu_0=nu_0, bell_height=bell_height, bell_width=bell_width, **kwargs)

        # Generate signals.
        phi = 2 * np.pi * np.dot(np.atleast_2d(nus).T, np.atleast_2d(t)).T
        
        if amplitudes is None:
            n_components = (l_max + 1) * (2 * k_max + 1)
            amplitudes = np.random.randn(2 * n_components)

        # TODO
        print("ANDY FIX THIS INCONSISTENCY IN API")

        amps = amplitudes * np.sqrt(amp_vars)

        fluxes = np.sum(
            amps[::2] * np.sin(phi) + amps[1::2] * np.cos(phi),
            axis=1)

        assert t.size == fluxes.size
        return fluxes


    def marginalized_log_likelihood(self, theta, t, residual_flux, ivar, 
        **kwargs):
        r"""
        Return the marginalized full log-likelihood of residual fluxes, given
        a model with parameters $\theta$.

        :param theta:
            The model parameters ($\nu_0$, $\Delta\nu$, $\nu_{\rm max}$,
            bell_height, bell_width, $r_{01}$).

        :param t:
            The observation times.

        :param residual_flux:
            The residual flux values for each observation, after subtracting the
            median flux.

        :param ivar:
            The inverse variances of the residual flux values for each
            observation.
        """

        nu_0, delta_nu, nu_max, bell_height, bell_width, r_01 = theta

        cov = np.eye(t.size) / ivar
        inv_cov = np.linalg.inv(cov)
        slogdet_cov = np.linalg.slogdet(2.0 * np.pi * cov)[1]

        nus, amp_vars = self.comb_frequencies(nu_max, delta_nu, nu_0=nu_0, 
            bell_height=bell_height, bell_width=bell_width, r_01=r_01, **kwargs)

        dm = _design_matrix(t, 2.0 * np.pi * nus)

        v_inv_mat, log_det = _update_inv_det(inv_cov, slogdet_cov, dm, amp_vars)
        ll = np.dot(residual_flux.T, np.dot(v_inv_mat, residual_flux)) + log_det
        return -0.5 * ll






def _design_matrix(t, omega):
    r"""
    Construct the design matrix.
    """

    design_matrix = np.zeros((len(t), 2 * len(omega)))
    design_matrix[:, ::2] = np.sin(omega * t)
    design_matrix[:, 1::2] = np.cos(omega * t)

    return design_matrix



def _update_inv_det(c_mat_inv, c_mat_log_det, b_mat, par_vars):

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


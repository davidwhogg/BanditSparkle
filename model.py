# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np


class AsteroseismicModel(object):

    """
    Forward-model light curves for giant stars with Solar-like oscillations.
    """

    def __init__(self, *args, **kwargs):
        return None


    def predict_power_spectrum(self, t, nu_max, delta_nu, **kwargs):
        r"""
        Predict the power spectrum for a given star. 

        Keyword arguments are passed directly to 
        :func:`AsteroseismicModel.predict_light_curve`.

        :param t:
            The times that the data were observed [seconds].

        :param nu_max:
            The frequency of maximum power [Hz].

        :param delta_nu:
            The large period separation [Hz].
        """

        # TODO HACK ASSUMES EVEN OBSERVATIONS
        omega = np.fft.rfftfreq(t.size, np.ptp(t)/t.size)
        f = np.fft.rfft(self.predict_light_curve(t, nu_max, delta_nu, **kwargs))
        return (omega, f)


    def comb_frequencies(self, nu_max, delta_nu, l_max=1, k_max=16, 
        r_01=0.5, amp_sigma=0.7, d_k_01=0.5, nu_0=None, bell_height=None, 
        bell_width=None, exposure_times=None, **kwargs):
        r"""
        Produce a comb of asteroseismic frequencies.

        :param t:
            The times that the data were observed [seconds].

        :param nu_max:
            The frequency of maximum power [Hz].

        :param delta_nu:
            The large period separation [Hz].

        :param l_max: [optional]
            The maximum number of l-modes to model (default: 1).

        :param k_max: [optional]
            The maximum number of k-modes to model (default: 16).

        :param r_01: [optional]
            The ratio of peak powers between zero-order and first-order modes
            (default: 0.5).

        :param amp_sigma: [optional]
            The intrinsic scatter in amplitudes (default: 0.7).

        :param d_k_01: [optional]
            The scale of spacing between small modes (default: 0.5).

        :param nu_0: [optional]
            The position of the strongest mode in the power spectrum [Hz].
            If ``None`` is given, then this will default to ``nu_max``.

        :param bell_height: [optional]
            The height of the power bell [Hz]. If ``None`` is given, then this
            defaults to ``amp_sigma**2``.

        :param bell_width: [optional]
            The width of the power bell [Hz]. If ``None`` is given, then this
            defaults to ``4 * delta_nu * nu_0``.
        """

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
        nus = nu_0 + ks * delta_nu # ERMERGERD DIFFERNET CONVENTION YOURE FIRED
        amp_vars = bell_height * r_01 ** els * \
                   np.exp(-0.5 * ((nus - nu_max) / bell_width) ** 2)
        amp_vars = np.repeat(amp_vars, 2)

        return (nus, amp_vars)


    def predict_light_curve(self, t, nu_max, delta_nu, l_max=1, k_max=16, 
        r_01=0.5, amp_sigma=0.7, d_k_01=0.5, nu_0=None, bell_height=None, 
        bell_width=None, exposure_times=None, **kwargs):
        r"""
        Predict the light curve for the given observation times.

        :param t:
            The times that the data were observed [seconds].

        :param nu_max:
            The frequency of maximum power [Hz].

        :param delta_nu:
            The large period separation [Hz].

        :param l_max: [optional]
            The maximum number of l-modes to model (default: 1).

        :param k_max: [optional]
            The maximum number of k-modes to model (default: 16).

        :param r_01: [optional]
            The ratio of peak powers between zero-order and first-order modes
            (default: 0.5).

        :param amp_sigma: [optional]
            The intrinsic scatter in amplitudes (default: 0.7).

        :param d_k_01: [optional]
            The scale of spacing between small modes (default: 0.5).

        :param nu_0: [optional]
            The position of the strongest mode in the power spectrum [Hz].
            If ``None`` is given, then this will default to ``nu_max``.

        :param bell_height: [optional]
            The height of the power bell [Hz]. If ``None`` is given, then this
            defaults to ``amp_sigma**2``.

        :param bell_width: [optional]
            The width of the power bell [Hz]. If ``None`` is given, then this
            defaults to ``4 * delta_nu * nu_0``.
        """

        if exposure_times is not None:
            raise NotImplementedError("exposure times not implemented yet")

        # Default values for nu_0, bell_width, bell_height
        nus, amp_vars = self.comb_frequencies(nu_max, delta_nu, l_max=l_max,
            k_max=k_max, r_01=r_01, amp_sigma=amp_sigma, d_k_01=d_k_01,
            nu_0=nu_0, bell_height=bell_height, bell_width=bell_width, **kwargs)

        # Generate signals.
        phi = 2 * np.pi * np.dot(np.atleast_2d(nus).T, np.atleast_2d(t)).T
        
        n_components = (l_max + 1) * (2 * k_max + 1)
        amplitudes = np.random.randn(2 * n_components)

        amps = amplitudes * np.sqrt(amp_vars)
        fluxes = np.sum(
            amps[::2] * np.sin(phi) + amps[1::2] * np.cos(phi),
            axis=1)

        assert t.size == fluxes.size
        return fluxes


    def __call__(self, *args, **kwargs):
        return self.predict_light_curve(self, *args, **kwargs)


    def _pre_compute_covariance_matrices(self, residual_flux_err):
        """
        Pre compute the inverse of the observed covariance matrix and the 
        sum of the log of the determinant of the observed covariance matrix in
        order to speed up repeated marginalized log likelihood calls using
        the same flux errors.

        :param residual_flux_err:
            An array of uncertainties on the residual fluxes.
        """

        residual_flux_err = np.atleast_1d(residual_flux_err)
        N = residual_flux_err.size
        assert N > 1

        var = residual_flux_err**2
        inv_cov = np.eye(N) / var
        slogdet_cov = np.sum(np.log(2.0 * np.pi * np.ones(N) * residual_flux_err**2))

        self._pre_computed_matrices = (inv_cov, slogdet_cov)
        return self._pre_computed_matrices


    def marginalized_log_likelihood(self, theta, t, residual_flux, 
        residual_flux_err=None, **kwargs):
        r"""
        Return the marginalized full log-likelihood of residual fluxes, given
        a model with parameters $\theta$.

        :param theta:
            The model parameters ($\nu_0$, $\Delta\nu$, $\nu_{\rm max}$,
            bell_height, bell_width, $r_{01}$).

        :param t:
            The times that the data were observed [seconds].

        :param residual_flux:
            The residual flux values for each observation, after subtracting the
            median flux [mags].

        :param residual_flux_err: [optional]
            The uncertainties in the residual fluxes from cadence photometry.
            If `None` is supplied, then it is assumed that the required
            matrices have been pre-computed using 
            :func:`AsteroseismicModel._pre_compute_covariance_matrices`,
            otherwise an exception will be raised.

        :raises ValueError:
            If no residual flux values are provided and the required matrices
            have not been pre-computed.
        """

        nu_0, delta_nu, nu_max, bell_height, bell_width, r_01 = theta
        nus, amp_vars = self.comb_frequencies(nu_max, delta_nu, nu_0=nu_0, 
            bell_height=bell_height, bell_width=bell_width, r_01=r_01, **kwargs)

        if residual_flux_err is None:
            # Use pre-computed matrices.
            try:
                inv_cov, slogdet_cov = self._pre_computed_matrices

            except AttributeError:
                raise ValueError("matrices must be pre-computed "
                                 "if no flux errors are supplied")

        else:
            inv_cov, slogdet_cov = self._pre_compute_covariance_matrices(
                residual_flux_err)

        dm = _design_matrix(t, 2.0 * np.pi * nus)
        v_inv_mat, log_det = _update_inv_det(inv_cov, slogdet_cov, dm, amp_vars)
        ll = np.dot(residual_flux.T, np.dot(v_inv_mat, residual_flux)) + log_det        
        return -0.5 * ll






def _design_matrix(t, omega):
    r"""
    Construct the design matrix.
    """

    _ = np.dot(np.atleast_2d(omega).T, np.atleast_2d(t)).T
    design_matrix = np.zeros((len(t), 2 * len(omega)))
    design_matrix[:, ::2] = np.sin(_)
    design_matrix[:, 1::2] = np.cos(_)

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

        c_inv_b.shape, a_mat_inv_lambda.shape))

    foo = np.dot(a_mat_inv_lambda, c_inv_b.T)

    out_inv = c_mat_inv - np.dot(c_inv_b, foo)


    # update determinant
    out_log_det = np.linalg.slogdet(a_mat)[1] + c_mat_log_det


    return out_inv, out_log_det



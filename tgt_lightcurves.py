import numpy as np
import matplotlib.pyplot as mp
import mod_lightcurves as ml
import kplr
import scipy.optimize as so

# plot settings
lw = 1.5
mp.rc('font', family = 'serif')
mp.rcParams['text.latex.preamble'] = [r'\boldmath']
mp.rcParams['axes.linewidth'] = lw
mp.rcParams['lines.linewidth'] = lw

# settings
target = 1162746 # 9787239
download = False
cn_realisation = True
s2d = 24.0 * 3600.0

# download data if not already done
if download:

	# frequencies are in muHz (1 muHz => period of 278 hours)
	stars = np.genfromtxt('apokasc_table_2.txt', dtype = None, \
						  skip_header = 27)
	for star in stars:
		kic = star[0]
		dnu = star[10]
		dnu_err = star[11]
		nu_max = star[12]
		nu_max_err = star[13]
		if kic == target:
			om_max_tgt = nu_max * 2.0e-6 * np.pi
			om_max_err_tgt = nu_max_err * 2.0e-6 * np.pi
			print nu_max, dnu

	# process lightcurves of chosen star
	client = kplr.API()
	star = client.star(target)
	lcs = star.get_light_curves(short_cadence = False)
	n_lc = len(lcs)
	i_lc, time = np.array([]), np.array([])
	flux, ferr = np.array([]), np.array([])
	for i in range(len(lcs)):
		with lcs[i].open() as f:
			hdu_data = f[1].data
			time_lc = hdu_data["time"]
			flux_lc = hdu_data["sap_flux"]
			ferr_lc = hdu_data["sap_flux_err"]
			time = np.concatenate((time, time_lc))
			flux = np.concatenate((flux, flux_lc))
			ferr = np.concatenate((ferr, ferr_lc))
			i_lc = np.concatenate((i_lc, \
								   np.ones(len(time_lc)) * i))
			
	# clean out NaNs and save to file
	i_lc = i_lc[~np.isnan(flux)]
	time = time[~np.isnan(flux)] * s2d
	ferr = ferr[~np.isnan(flux)]
	flux = flux[~np.isnan(flux)]
	np.savetxt('test_{:d}_full_lightcurve.txt'.format(target), \
			   np.column_stack((i_lc, time.flatten(), \
			   					flux.flatten(), ferr.flatten())), \
			   fmt = '%3d %19.12e %19.12e %19.12e')
	print 'saved!'

else:

	# read data from file
	data = np.genfromtxt('test_{:d}_full_lightcurve.txt'.format(target))
	i_lc = data[:, 0]
	time = data[:, 1]
	flux = data[:, 2]
	ferr = data[:, 3]

# pick out good section of data
time = time[i_lc == 3]
ferr = ferr[i_lc == 3]
flux = flux[i_lc == 3]

'''
time = time[0:2000]
ferr = ferr[0:2000]
flux = flux[0:2000]
'''

# if cutting by time, reduce poly fit order from 8 to 6
#ferr = ferr[time < np.min(time) + 30 * s2d]
#flux = flux[time < np.min(time) + 30 * s2d]
#time = time[time < np.min(time) + 30 * s2d]

# characterize data segment
noise_var = np.mean(ferr ** 2)
n_samples = len(time)
duration = (time[-1] - time[0]) / s2d
print '{:d} samples spanning {:.2f} d'.format(n_samples, \
											  duration)
print 'diagonal RMS noise: {:8.2e}'.format(np.sqrt(noise_var))

# fit polynomial to remove long-timescale variability. plot results
order = 8
pfit = np.polyfit(time, flux, order, w = 1.0 / ferr)
poly = np.poly1d(pfit)
bc_flux = flux - poly(time)
mp.plot(time / s2d, flux)
mp.fill_between(time / s2d, flux, flux + ferr, color = 'grey')
mp.fill_between(time / s2d, flux, flux - ferr, color = 'grey')
mp.plot(time / s2d, poly(time), 'k')
mp.xlabel(r'$t\,[{\rm d}]$')
mp.ylabel(r'${\rm flux}$')
mp.show()

# calculate auto-correlation in order to estimate coloured noise.
# fit auto-correlation with exponential decay. this is specific to 
# the current target
autocorr = ml.autocorrelate(flux)
bc_autocorr = ml.autocorrelate(bc_flux)
lag = (-n_samples + 1 + np.arange(2 * n_samples - 1)) * \
	  (time[1] - time[0])
#i_fit = (lag > 0.0) & (lag < 3.0)
i_fit = (np.abs(lag) < 3.0 * s2d)
p_ini = [np.max(bc_autocorr[i_fit]), 24.0 * 3600.0]
p_opt, p_cov = so.curve_fit(ml.laplace, lag[i_fit], \
							bc_autocorr[i_fit], p_ini)
np.savetxt('test_{:d}_autocorr_fit.txt'.format(target), \
		   p_opt, fmt='%19.12e')

# optionally generate noise realization from k_mat
if cn_realisation:
	k_mat = ml.col_noise_cov_mat(time, *p_opt)
	cn = ml.col_noise(k_mat)
	cn_autocorr = ml.autocorrelate(cn)

# plot!
if cn_realisation:
	fig, axes = mp.subplots(3, 2, figsize=(16, 15))
else:
	fig, axes = mp.subplots(2, 2, figsize=(16, 10))
axes[0, 0].plot(time / s2d, flux)
axes[0, 0].plot(time / s2d, poly(time), 'k')
axes[0, 0].set_xlabel('time [d]')
axes[0, 0].set_ylabel('flux')
axes[0, 1].plot(lag / s2d, autocorr)
axes[0, 1].set_xlabel('lag [d]')
axes[0, 1].set_ylabel('autocorrelation')
axes[1, 0].plot(time / s2d, bc_flux)
axes[1, 0].set_xlabel('time [d]')
axes[1, 0].set_ylabel('BS subtracted flux')
axes[1, 1].plot(lag / s2d, bc_autocorr)
axes[1, 1].set_xlabel('lag [d]')
axes[1, 1].set_ylabel('BS subtracted autocorrelation')
if cn_realisation:
	axes[2, 0].plot(time / s2d, cn)
	axes[2, 0].set_xlabel('time [d]')
	axes[2, 0].set_ylabel('coloured noise flux')
	axes[2, 1].plot(lag / s2d, cn_autocorr)
	axes[2, 1].set_xlabel('lag [d]')
	axes[2, 1].set_ylabel('coloured noise autocorrelation')
mp.savefig('test_{:d}_autocorr.pdf'.format(target), \
		   bbox_inches='tight')
mp.show()
mp.plot(lag[i_fit] / s2d, ml.laplace(lag[i_fit], *p_opt))
mp.plot(lag[i_fit] / s2d, bc_autocorr[i_fit])
if cn_realisation:
	mp.plot(lag[i_fit] / s2d, cn_autocorr[i_fit])
mp.xlabel('lag [d]')
mp.ylabel('BS subtracted autocorrelation')
mp.savefig('test_{:d}_autocorr_fit.pdf'.format(target), \
		   bbox_inches='tight')
mp.show()

# save baseline-corrected lightcurve to file
# @TODO: flux errors aren't retrieved with same accuracy as other two fields
np.savetxt('test_{:d}_lightcurve.txt'.format(target), \
		   np.column_stack((time.flatten(), bc_flux.flatten(), \
		   					ferr.flatten())), \
		   fmt = '%19.12e')

# interpolate to allow Fourier transform. it's important to note a 
# few things about Fourier transform conventions here. Numpy's 
# normalization is 1 for the forward transform and 1/N for the 
# inverse. to obtain the correct units, we need to multiply each 
# forward transform by 1/N and each inverse by N. in addition:
# 1) Numpy's correlate function does not take the expectation value.
#    I therefore wrap it to divide by n_samples, yielding a variance.
# 2) The PSD is the Fourier transform of the autocorrelation (as I 
#    have defined it, i.e. including the expectation value).
# 3) The PSD is simply |FT|^2
# 4) I'm fitting the autocorrelation with a Laplace function (i.e., 
#    exponential decay: a exp(-|lag|/b)). This has a simple 
#    *continuous* Fourier transform: 2a/b/(1/b**2+w**2); in order to 
#    bring this inline with our conventions we must divide by dt and
#    n_samples
tint = np.linspace(time[0], time[-1], n_samples)
dt = tint[1] - tint[0]
fint = np.interp(tint, time, bc_flux)
ft = np.fft.rfft(fint) / n_samples
psd = ft * ft.conj()
if cn_realisation:
	fint_cn = np.interp(tint, time, cn)
	ft_cn = np.fft.rfft(fint_cn) / n_samples
	psd_cn = ft_cn * ft_cn.conj()
om = 2.0 * np.pi * \
	 np.fft.rfftfreq(n_samples, tint[1] - tint[0])
bc_psd = 2.0 * p_opt[0] / p_opt[1] / \
		 (1.0 / p_opt[1] ** 2 + om ** 2) / dt / n_samples
env_psd = 400.0 * \
		  np.exp(-0.5 * ((om / 2.0 / np.pi - 27.7e-6) / 3.95e-6) ** 2)
#env_psd = 1500.0 * \
#		  np.exp(-0.5 * ((om / 2.0 / np.pi - 2.83446845e-05) / 3.93923782e-05) ** 2)
'''
mp.plot((tint - (tint[0] + dt * n_samples / 2)) / s2d, \
		np.roll(np.fft.irfft(bc_psd) * n_samples, n_samples / 2))
mp.xlabel(r'$\tau [{\rm d}]$')
mp.ylabel(r'${\rm autocorrelation}$')
mp.show()
'''
fig, axes = mp.subplots(1, 2, figsize=(16,5))
axes[0].step(tint / s2d, fint, color='k')
if cn_realisation:
	axes[0].step(tint / s2d, fint_cn, color='g', alpha=0.5)
axes[0].set_xlabel(r'$t\,[{\rm d}]$')
axes[0].set_ylabel(r'baseline-corrected flux $[s^{-1}]$')
axes[1].loglog(om * 1e6, psd, color='k')
if cn_realisation:
	axes[1].loglog(om * 1e6, psd_cn, color='g', alpha=0.5)
axes[1].plot(om * 1e6, bc_psd, color='r')
axes[1].plot(om * 1e6, bc_psd + env_psd, color='orange')
axes[1].axhline(noise_var / n_samples, color='gray', ls='--')
axes[1].set_xlim(np.min(om) * 1e6, np.max(om) * 1e6)
axes[1].set_xlabel(r'$\omega [{\rm \mu Hz}]$')
axes[1].set_ylabel(r'power spectrum $[s^{-2}]$')
mp.savefig('test_{:d}_input_plots.pdf'.format(target), \
		   bbox_inches='tight')
mp.show()

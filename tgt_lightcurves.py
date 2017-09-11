import numpy as np
import matplotlib.pyplot as mp
import mod_lightcurves as ml
import kplr
import scipy.optimize as so

# settings
target = 1162746 # 9787239
download = False
cn_realisation = True

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
	time = time[~np.isnan(flux)]
	ferr = ferr[~np.isnan(flux)]
	flux = flux[~np.isnan(flux)]
	np.savetxt('test_{:d}_full_lightcurve.txt'.format(target), \
			   np.column_stack((i_lc, time.flatten() * 24.0 * 3600.0, \
			   					flux.flatten(), ferr.flatten())), \
			   fmt = '%3d %19.12e %19.12e %19.12e')
	print 'saved!'

else:

	# read data from file
	data = np.genfromtxt('test_{:d}_full_lightcurve.txt'.format(target))
	i_lc = data[:, 0]
	time = data[:, 1] / 24.0 / 3600.0
	flux = data[:, 2]
	ferr = data[:, 3]

# pick out good section of data
time = time[i_lc == 3]
ferr = ferr[i_lc == 3]
flux = flux[i_lc == 3]

# if cutting by time, reduce poly fit order from 8 to 6
#ferr = ferr[time < np.min(time) + 30]
#flux = flux[time < np.min(time) + 30]
#time = time[time < np.min(time) + 30]

# characterize data segment
noise_var = np.mean(ferr ** 2)
n_samples = len(time)
duration = time[-1] - time[0]
print '{:d} samples spanning {:.2f} d'.format(n_samples, \
											  duration)
print np.sqrt(noise_var)

# fit polynomial to remove long-timescale variability. plot results
order = 8
pfit = np.polyfit(time, flux, order, w = 1.0 / ferr)
poly = np.poly1d(pfit)
bc_flux = flux - poly(time)
mp.plot(time, flux)
mp.fill_between(time, flux, flux + ferr, color = 'grey')
mp.fill_between(time, flux, flux - ferr, color = 'grey')
mp.plot(time, poly(time), 'k')
mp.show()

# calculate auto-correlation in order to estimate coloured noise.
# fit auto-correlation with exponential decay. this is specific to 
# the current target
autocorr = ml.autocorrelate(flux)
bc_autocorr = ml.autocorrelate(bc_flux)
lag = (-n_samples + 1 + np.arange(2 * n_samples - 1)) * \
	  (time[1] - time[0])
#i_fit = (lag > 0.0) & (lag < 3.0)
i_fit = (np.abs(lag) < 3.0)
p_ini = [bc_autocorr[i_fit][0], 1.0]
p_opt, p_cov = so.curve_fit(ml.laplace, lag[i_fit], \
							bc_autocorr[i_fit], p_ini)
np.savetxt('test_{:d}_autocorr_fit.txt'.format(target), \
		   [p_opt[0], p_opt[1] * 24.0 * 3600.0], \
		   fmt='%19.12e')

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
axes[0, 0].plot(time, flux)
axes[0, 0].plot(time, poly(time), 'k')
axes[0, 0].set_xlabel('time [d]')
axes[0, 0].set_ylabel('flux')
axes[0, 1].plot(lag, autocorr)
axes[0, 1].set_xlabel('lag [d]')
axes[0, 1].set_ylabel('autocorrelation')
axes[1, 0].plot(time, bc_flux)
axes[1, 0].set_xlabel('time [d]')
axes[1, 0].set_ylabel('BS subtracted flux')
axes[1, 1].plot(lag, bc_autocorr)
axes[1, 1].set_xlabel('lag [d]')
axes[1, 1].set_ylabel('BS subtracted autocorrelation')
if cn_realisation:
	axes[2, 0].plot(time, cn)
	axes[2, 0].set_xlabel('time [d]')
	axes[2, 0].set_ylabel('coloured noise flux')
	axes[2, 1].plot(lag, cn_autocorr)
	axes[2, 1].set_xlabel('lag [d]')
	axes[2, 1].set_ylabel('coloured noise autocorrelation')
mp.savefig('test_{:d}_autocorr.pdf'.format(target), \
		   box_inches='tight')
mp.show()
mp.plot(lag[i_fit], ml.laplace(lag[i_fit], *p_opt))
mp.plot(lag[i_fit], bc_autocorr[i_fit])
if cn_realisation:
	mp.plot(lag[i_fit], cn_autocorr[i_fit])
mp.xlabel('lag [d]')
mp.ylabel('BS subtracted autocorrelation')
mp.savefig('test_{:d}_autocorr_fit.pdf'.format(target), \
		   box_inches='tight')
mp.show()

# save baseline-corrected lightcurve to file
# @TODO: flux errors aren't retrieved with same accuracy as other two fields
np.savetxt('test_{:d}_lightcurve.txt'.format(target), \
		   np.column_stack((time.flatten() * 24.0 * 3600.0, \
		   					bc_flux.flatten(), ferr.flatten())), \
		   fmt = '%19.12e')

# interpolate to allow Fourier transform
tint = np.linspace(time[0], time[-1], n_samples)
fint = np.interp(tint, time, bc_flux)
ft = np.fft.rfft(fint)
if cn_realisation:
	fint_cn = np.interp(tint, time, cn)
	ft_cn = np.fft.rfft(fint_cn)
om = 2.0 * np.pi * \
	 np.fft.rfftfreq(n_samples, (tint[1] - tint[0]) * 24.0 * 3600.0)
mp.rcParams["figure.figsize"] = [8, 8]
fig, axes = mp.subplots(2)
axes[0].step(tint, fint)
axes[0].set_xlabel(r'$t\,[{\rm d}]$')
axes[0].set_ylabel(r'baseline-corrected flux $[s^{-1}]$')
axes[1].loglog(om * 1e6, np.absolute(ft) ** 2)
if cn_realisation:
	axes[1].loglog(om * 1e6, np.absolute(ft_cn) ** 2, '--', alpha=0.5)
#axes[1].axvspan((om_max_tgt - om_max_err_tgt) * 1e6, \
#				(om_max_tgt + om_max_err_tgt) * 1e6, \
#				color = 'red', alpha = 0.5)
axes[1].axhline(n_samples * noise_var, color = 'gray', ls = '--')
axes[1].set_xlim(np.min(om) * 1e6, np.max(om) * 1e6)
axes[1].set_xlabel(r'$\omega [{\rm \mu Hz}]$')
axes[1].set_ylabel(r'power spectrum $[s^{-2}]$')
mp.savefig('test_{:d}_input_plots.pdf'.format(target), box_inches = 'tight')
mp.show()

import numpy as np
import matplotlib.pyplot as mp
import matplotlib.cm as mpcm
import matplotlib.colors as mpc
import mod_lightcurves as ml
import kplr
import scipy.optimize as so
import scipy.signal as ss
import astropy.stats as aps
import random as ra

def running_mean(x, n):
    
    # @TODO: check n odd?
    # pad input array and calculate cumsum
    n_out = len(x)
    n_pad = (n - 1) / 2
    padded = np.insert(np.insert(x, len(x), np.zeros(n_pad)), \
                       0, np.zeros(n_pad))
    cumsum = np.zeros(n_out)
    for i in range(n_out):
        cumsum[i] = np.sum(padded[i: i + 2 * n_pad + 1])

    # adjust denominator in mean and return
    n_in_sum = np.ones(n_out) * n
    for i in range(n_pad):
        n_in_sum[i] = n - n_pad + i
        n_in_sum[-i-1] = n - n_pad + i
    return cumsum / n_in_sum

# plot settings
lw = 1.5
mp.rc('font', family = 'serif')
mp.rcParams['text.latex.preamble'] = [r'\boldmath']
mp.rcParams['axes.linewidth'] = lw
mp.rcParams['lines.linewidth'] = lw
cm = mpcm.get_cmap('plasma')

# settings
#target = 1162746
#target = 6757558 # good!
#target = 6104786
target = 7674224 # good!
#target = 6779699 # might be quite a lot of excess high-freq noise
download = True
use_cleaned_flux = True # use PDCSAP_FLUX
cn_realisation = True
s2d = 24.0 * 3600.0
compare_psd_estimators = False
n_samples_max = 1000 # TESS = 1440
random_time_sampling = False

# pick out interesting stars, along with the features of the target.
# frequencies are in muHz (1 muHz => period of 278 hours)
kic_ids = []
dnus = []
dnu_sigs = []
dnu_n_sigs = []
nu_maxes = []
nu_max_sigs = []
nu_max_n_sigs = []
stars = np.genfromtxt('apokasc_table_2.txt', dtype = None, \
                      skip_header = 27)
i_tgt = None
for i in range(len(stars)):
    star = stars[i]
    kic_ids.append(star[0])
    dnus.append(star[10])
    dnu_sigs.append(star[11])
    dnu_n_sigs.append(star[10] / star[11])
    nu_maxes.append(star[12])
    nu_max_sigs.append(star[13])
    nu_max_n_sigs.append(star[12] / star[13])
    if target == kic_ids[-1]:
        i_tgt = i
combo_n_sigs = np.array(dnu_n_sigs) + np.array(nu_max_n_sigs)
i_sort = np.argsort(combo_n_sigs)
print 'top APOKASC parameter significances:' 
print '#  id       dnu   nu_m'
for i in range(1, 11):
    ind = i_sort[-i]
    fmt = '{:2d} {:8d} {:5.2f} {:5.2f}'
    print fmt.format(i, kic_ids[ind], dnu_n_sigs[ind], \
                     nu_max_n_sigs[ind])
print 'target: ', target
if i_tgt is None:
    print 'target not found in apokasc...'
else:
    print 'target dnu = {:10.4e} muHz ({:5.2f}-sigma)'.format(dnus[i_tgt], \
                                                              dnu_n_sigs[i_tgt])
    print 'target nu_max = {:10.4e} muHz ({:5.2f}-sigma)'.format(nu_maxes[i_tgt], \
                                                                 nu_max_n_sigs[i_tgt])
    dnu_tgt = dnus[i_tgt] * 1e-6
    nu_max_tgt = nu_maxes[i_tgt] * 1e-6
    dnu_sig_tgt = dnu_sigs[i_tgt] * 1e-6
    nu_max_sig_tgt = nu_max_sigs[i_tgt] * 1e-6
if target == 1162746:
    h_tgt = 340.0
    w_tgt = 10.0 * 1e-6
elif target == 6757558:
    h_tgt = 4.0
    w_tgt = 10.0 * 1e-6
elif target == 6104786:
    h_tgt = 20.0
    w_tgt = 10.0 * 1e-6
elif target == 9778288:
    h_tgt = 20.0
    w_tgt = 5.0 * 1e-6
elif target == 7674224:
    h_tgt = 20.0
    w_tgt = 5.0 * 1e-6
elif target == 6779699:
    dnu_tgt = 8.0e-6
    nu_max_tgt = 90.0e-6 #roughly!
    nu_max_sig_tgt = 5.0e-6
    h_tgt = 20.0
    w_tgt = 5.0 * 1e-6

# download data if not already done
if download:

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
            if use_cleaned_flux:
                flux_lc = hdu_data["pdcsap_flux"]
                ferr_lc = hdu_data["pdcsap_flux_err"]
            else:
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


# loop through data segments to estimate PSDs
nu_min, nu_max = nu_max_tgt - 3 * dnu_tgt, nu_max_tgt + 3 * dnu_tgt
fig, axes = mp.subplots(1, 2, figsize=(16,5))
if target == 1162746:
    ap_nu = np.linspace(nu_min, nu_max, 950)
elif target == 6757558:
    ap_nu = np.linspace(nu_min, nu_max, 2850)
elif target == 6104786:
    ap_nu = np.linspace(nu_min, nu_max, 2850)
elif target == 9778288:
    ap_nu = np.linspace(nu_min, nu_max, 2850)
elif target == 7674224:
    ap_nu = np.linspace(nu_min, nu_max, 2850)
elif target == 6779699:
    ap_nu = np.linspace(nu_min, nu_max, 2850)
mean_ls = np.zeros(len(ap_nu))
mean_psd = np.zeros(2**10 + 1)
if use_cleaned_flux:
    order = 0
else:
    order = 10 # 8
n_lc = int(np.max(i_lc) + 1)
lc_col = np.arange(n_lc) / float(n_lc - 1)
for i in range(n_lc):

    # shenanigans
    inds = (i_lc==i)
    n_inds = np.sum(inds)
    if compare_psd_estimators:
        col = cm(0.0)
    else:
        col = cm(lc_col[i])

    # fit out long-timescale variability. if using cleaned PDCSAP_FLUX
    # (http://archive.stsci.edu/kepler/manuals/archive_manual.pdf), 
    # order = 0 so just fitting out mean
    pfit = np.polyfit(time[inds], flux[inds], order, w = 1.0 / ferr[inds])
    poly = np.poly1d(pfit)
    bc_flux = flux[inds] - poly(time[inds])
    try:
        long_bc_flux
    except NameError:
        long_bc_flux = bc_flux
    else:
        long_bc_flux = np.concatenate((long_bc_flux, bc_flux))
    print i, n_inds
    axes[0].plot(time[inds] / s2d, flux[inds], color=col)

    # calculate LS periodogram of segment data
    ap_ls = aps.LombScargle(time[inds], bc_flux).power(ap_nu, normalization='psd')
    mean_ls += ap_ls
    #ap_nu, ap_ls = aps.LombScargle(time[inds], bc_flux).autopower(minimum_frequency=nu_min, maximum_frequency=nu_max, normalization='psd')
    #print len(ap_nu), ap_nu.min(), ap_nu.max()
    plot_nu = (ap_nu >= nu_min) & (ap_nu <= nu_max)
    axes[1].plot(ap_nu[plot_nu] * 1.0e6, ap_ls[plot_nu], color=col, label='LS')

    # calculate a couple of other estimators that rely on regularly
    # sampled data
    if n_inds > 2 ** 11:

        # interpolate segment onto regular grid to check FT-based 
        # methods
        t_sample = 1800.0
        f_sample = 1.0 / t_sample
        n_sample_int = int(np.ceil((time[inds][-1] - time[inds][0]) / \
                                   t_sample))
        t_int = time[inds][0] + np.arange(n_sample_int) * t_sample
        f_int = np.interp(t_int, time[inds], flux[inds])

        # first check out Welch's method, which averages short sub-
        # segments
        ss_nu, ss_psd = ss.welch(f_int, f_sample, nperseg=2**11)
        ss_psd /= n_sample_int
        mean_psd += ss_psd

        # also look at full-length FFT
        np_ft = np.fft.rfft(f_int) / n_sample_int
        np_psd = np_ft * np_ft.conj() * n_sample_int
        np_nu = np.fft.rfftfreq(n_sample_int, t_sample)

        # quick comparison of estimators
        if compare_psd_estimators:
            plot_nu = (ss_nu >= nu_min) & (ss_nu <= nu_max)
            axes[1].plot(ss_nu[plot_nu], ss_psd[plot_nu], \
                         color=cm(0.45), label='Welch')
            plot_nu = (np_nu >= nu_min) & (np_nu <= nu_max)
            axes[1].plot(np_nu[plot_nu], np_psd[plot_nu], \
                         color=cm(0.9), label='FFT')
            axes[1].legend(loc='upper left')
            axes[1].set_xlim(nu_min, nu_max)
            mp.show()
            exit()

# finish off plot of estimates from disjoint lightcurves
axes[1].axvline(nu_max_tgt * 1.0e6, color='k', ls='--')
axes[1].axvline((nu_max_tgt + dnu_tgt) * 1.0e6, color='k', ls='-.')
axes[1].axvline((nu_max_tgt - dnu_tgt) * 1.0e6, color='k', ls='-.')
axes[1].axvspan((nu_max_tgt - nu_max_sig_tgt) * 1.0e6, \
                (nu_max_tgt + nu_max_sig_tgt) * 1.0e6, \
                color='k', alpha=0.25, zorder=-1)
axes[1].set_xlim(nu_min * 1.0e6, nu_max * 1.0e6)
axes[1].set_xlabel('frequency [$\mu$Hz]')
axes[1].set_ylabel('PSD')
axes[0].set_xlim(np.min(time) / s2d, np.max(time) / s2d)
axes[0].set_xlabel('time [d]')
axes[0].set_ylabel('flux')
#axes[1].set_ylim(0, 7e6)
#axes[1].set_ylim(0, 5e4)
mp.show()

# tidy up means and calculate PSD of complete timeseries
ap_ls = aps.LombScargle(time, long_bc_flux).power(ap_nu, normalization='psd')
long_ap_nu, long_ap_ls = aps.LombScargle(time, long_bc_flux).autopower(minimum_frequency=nu_min, maximum_frequency=nu_max, normalization='psd')
print len(long_ap_nu)
mean_ls /= n_lc
mean_psd /= n_lc
long_ap_ls_s = running_mean(long_ap_ls, 21)

# summary
plot_smooth = True
fig, axes = mp.subplots(1, 2, figsize=(16,5))
axes[0].plot(time / s2d, long_bc_flux, 'k')
axes[0].set_xlim(np.min(time) / s2d, np.max(time) / s2d)
if plot_smooth:
    axes[1].plot(long_ap_nu * 1.0e6, long_ap_ls_s, color=cm(0.7), \
                  label='full LSP (smoothed)')
else:
    axes[1].plot(long_ap_nu * 1.0e6, long_ap_ls, color=cm(0.7), \
                 label='full LSP')
axes[1].plot(ap_nu * 1.0e6, mean_ls, color=cm(0), \
             label='mean segment LSP')
axes[1].plot(ss_nu * 1.0e6, mean_psd, color=cm(0.35), \
             label='mean segment PSD')
axes[1].axvline(nu_max_tgt * 1.0e6, color='k', ls='--', \
                alpha=1.0, zorder=0)
axes[1].axvline((nu_max_tgt + dnu_tgt) * 1.0e6, color='k', ls='-.', \
                alpha=1.0, zorder=0)
axes[1].axvline((nu_max_tgt - dnu_tgt) * 1.0e6, color='k', ls='-.', \
                alpha=1.0, zorder=0)
axes[1].axvspan((nu_max_tgt - nu_max_sig_tgt) * 1.0e6, \
                (nu_max_tgt + nu_max_sig_tgt) * 1.0e6, color='k', \
                alpha=0.25, zorder=-1)
i_max = mean_ls.argmax()
axes[1].axvline(ap_nu[i_max] * 1.0e6, color=cm(0), ls='--', \
                alpha=0.5, zorder=0)
axes[1].axvline((ap_nu[i_max] + dnu_tgt) * 1.0e6, color=cm(0), \
                ls='-.', alpha=0.5, zorder=0)
axes[1].axvline((ap_nu[i_max] - dnu_tgt) * 1.0e6, color=cm(0), \
                ls='-.', alpha=0.5, zorder=0)
i_max = ap_ls.argmax()
axes[1].axvline(ap_nu[i_max] * 1.0e6, color=cm(0.7), ls='--', \
                alpha=0.5, zorder=0)
axes[1].axvline((ap_nu[i_max] + dnu_tgt) * 1.0e6, color=cm(0.7), \
                ls='-.', alpha=0.5, zorder=0)
axes[1].axvline((ap_nu[i_max] - dnu_tgt) * 1.0e6, color=cm(0.7), \
                ls='-.', alpha=0.5, zorder=0)
axes[1].set_xlim(nu_min * 1.0e6, nu_max * 1.0e6)
if plot_smooth:
    axes[1].set_ylim(0, 1.05 * long_ap_ls_s.max())
else:
    axes[1].set_ylim(0, 1.05 * long_ap_ls.max())
axes[1].legend(loc='upper left', fontsize=10)
mp.show()

# pick out good section of data
if target == 1162746:
    i_lc_tgt = 3
elif target == 6757558:
    i_lc_tgt = 9
elif target == 6104786:
    i_lc_tgt = 4
elif target == 9778288:
    i_lc_tgt = 6
elif target == 7674224:
    i_lc_tgt = 5 # 4 # 9
elif target == 6779699:
    i_lc_tgt = 4
time = time[i_lc == i_lc_tgt]
ferr = ferr[i_lc == i_lc_tgt]
flux = flux[i_lc == i_lc_tgt]

# downsample if desired, optionally randomizing samples first 
if n_samples_max is not None:
	if random_time_sampling:
		rts_inds = range(len(time))
		ra.shuffle(rts_inds)
		rts_inds = np.sort(rts_inds[0: n_samples_max])
		rts_time = time[rts_inds]
		rts_ferr = ferr[rts_inds]
		rts_flux = flux[rts_inds]
	time = time[0: n_samples_max]
	ferr = ferr[0: n_samples_max]
	flux = flux[0: n_samples_max]

'''
# if cutting by time, reduce poly fit order from 8 to 6
time = time[0:2000]
ferr = ferr[0:2000]
flux = flux[0:2000]
ferr = ferr[time < np.min(time) + 30 * s2d]
flux = flux[time < np.min(time) + 30 * s2d]
time = time[time < np.min(time) + 30 * s2d]
'''

# characterize data segment
noise_var = np.mean(ferr ** 2)
n_samples = len(time)
duration = (time[-1] - time[0]) / s2d
print '{:d} samples spanning {:.2f} d'.format(n_samples, \
                                              duration)
print 'diagonal RMS noise: {:8.2e}'.format(np.sqrt(noise_var))

# fit polynomial to remove long-timescale variability. plot results
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
if target == 1162746:
	p_opt[0] *= 3.5

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
if random_time_sampling:
	rts_pfit = np.polyfit(rts_time, rts_flux, order, w = 1.0 / rts_ferr)
	rts_poly = np.poly1d(rts_pfit)
	rts_bc_flux = rts_flux - rts_poly(rts_time)
	#mp.plot(time, bc_flux)
	#mp.plot(rts_time, rts_bc_flux)
	#mp.show()
	np.savetxt('test_{:d}_lightcurve.txt'.format(target), \
	           np.column_stack((rts_time.flatten(), \
	           					rts_bc_flux.flatten(), \
	                            rts_ferr.flatten())), \
	           fmt = '%19.12e')
else:
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

# calculate expected PSDs of noise and Gaussian envelope
bc_psd = 2.0 * p_opt[0] / p_opt[1] / \
         (1.0 / p_opt[1] ** 2 + om ** 2) / dt / n_samples
env_psd = h_tgt * \
          np.exp(-0.5 * ((om / 2.0 / np.pi - nu_max_tgt) / dnu_tgt) ** 2)

# plot!
if False:
    plot_freq = om * 1.0e6
    plot_freq_lab = r'$\omega [{\rm \mu Hz}]$'
else:
    plot_freq = om * 1.0e6 / 2.0 / np.pi
    plot_freq_lab = r'$\nu [{\rm \mu Hz}]$'
fig, axes = mp.subplots(1, 2, figsize=(16,5))
axes[0].step(tint / s2d, fint, color='k')
if cn_realisation:
    axes[0].step(tint / s2d, fint_cn, color='g', alpha=0.5)
axes[0].set_xlabel(r'$t\,[{\rm d}]$')
axes[0].set_ylabel(r'baseline-corrected flux $[s^{-1}]$')
axes[1].loglog(plot_freq, psd, color='k', label='data')
if cn_realisation:
    axes[1].loglog(plot_freq, psd_cn, color='g', alpha=0.5, \
                   label='noise realization')
axes[1].plot(plot_freq, bc_psd, color='r', label='envelope model')
axes[1].plot(plot_freq, bc_psd + env_psd, color='orange', \
             label='noise model')
axes[1].axhline(noise_var / n_samples, color='gray', ls='--')
axes[1].set_xlim(np.min(plot_freq), np.max(plot_freq))
axes[1].set_xlabel(plot_freq_lab)
axes[1].set_ylabel(r'power spectrum $[s^{-2}]$')
axes[1].legend(loc='lower left')
mp.savefig('test_{:d}_input_plots.pdf'.format(target), \
           bbox_inches='tight')
mp.show()

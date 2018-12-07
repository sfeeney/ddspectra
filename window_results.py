import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import scipy.stats as sps
import scipy.linalg.lapack as spl
import scipy.special as spesh
import itertools as it
import h5py
import os
if 'DISPLAY' not in os.environ.keys():
	import matplotlib
	matplotlib.use('Agg')
import matplotlib.pyplot as mp
import matplotlib.cm as mpcm
import matplotlib.axes as mpa

def axis_to_axes(axis, transpose=False):
	if isinstance(axis, mpa.Axes):
		axes = np.empty(shape=(1), dtype=mpa.Axes)
		axes[0] = axis
	else:
		if transpose:
			axes = np.empty(shape=(len(axis), 1), dtype=object)
			axes[:, 0] = axis
		else:
			axes = np.empty(shape=(1, len(axis)), dtype=object)
			axes[0, :] = axis
	return axes

def pretty_hist(data, bins, axis, color, density=False, fill=True, \
				ls='-', zorder=None):

	hist, bin_edges = np.histogram(data, bins=bins, density=density)
	bins_to_plot = np.append(bins, bins[-1])
	hist_to_plot = np.append(np.insert(hist, 0, 0.0), 0.0)
	if zorder is not None:
		axis.step(bins_to_plot, hist_to_plot, where='pre', \
				  color=color, linestyle=ls, zorder=zorder)
		if fill:
			axis.fill_between(bins_to_plot, hist_to_plot, \
							  color=color, alpha=0.7, step='pre', \
							  zorder=zorder)
	else:
		axis.step(bins_to_plot, hist_to_plot, where='pre', \
				  color=color, linestyle=ls)
		if fill:
			axis.fill_between(bins_to_plot, hist_to_plot, \
							  color=color, alpha=0.7, step='pre')

def read_spectra(n_to_load, filename, return_wl=False):

	try:
		f = h5py.File(filename, 'r')
	except:
		msg = 'ERROR: input file ' + filename + ' not found'
		if use_mpi:
			print msg
			mpi.COMM_WORLD.Abort()
		else:
			exit(msg)
	file_data = f['dataset_1'][:]
	n_in_file = file_data.shape[1]
	to_load = np.arange(n_in_file) < n_to_load
	if return_wl:
		return file_data[:, 0, 0], file_data[:, to_load, 1:]
	else:
		return file_data[:, to_load, 1:]

def split_covariance(i_cond, cov, wendices, include=None, i_pred=None):

	# find indices. predicting inds_pred conditioned on inds_cond. 
	# get conditional indices first
	wl = np.arange(cov.shape[0])
	if isinstance(i_cond, list):
		inds_cond = np.full(len(wl), False, dtype=bool)
		for i_c in i_cond:
			if i_c == 0:
				inds_cond = inds_cond | (wl < wendices[i_c])
			else:
				inds_cond = inds_cond | ((wl >= wendices[i_c - 1]) & \
							(wl < wendices[i_c]))
	else:
		if i_cond == 0:
			inds_cond = (wl < wendices[i_cond])
		else:
			inds_cond = (wl >= wendices[i_cond - 1]) & \
						(wl < wendices[i_cond])
	n_cond = np.sum(inds_cond)

	# now find indices to predict
	if i_pred is None:
		inds_pred = ~inds_cond
	else:
		if isinstance(i_pred, list):
			inds_pred = np.full(len(wl), False, dtype=bool)
			for i_p in i_pred:
				if i_p == 0:
					inds_pred = inds_pred | (wl < wendices[i_p])
				else:
					inds_pred = inds_pred | (wl >= wendices[i_p - 1]) & \
								(wl < wendices[i_p])
		else:
			if i_pred == 0:
				inds_pred = (wl < wendices[i_pred])
			else:
				inds_pred = (wl >= wendices[i_pred - 1]) & \
							(wl < wendices[i_pred])
	n_pred = np.sum(inds_pred)

	# construct submatrices
	s_cc = np.zeros((n_cond, n_cond))
	s_cp = np.zeros((n_cond, n_pred))
	s_pp = np.zeros((n_pred, n_pred))
	n = 0
	m = 0
	for j in range(cov.shape[0]):
		if inds_cond[j]:
			s_cc[n, :] = cov[j, inds_cond]
			s_cp[n, :] = cov[j, inds_pred]
			n += 1
		if inds_pred[j]:
			s_pp[m, :] = cov[j, inds_pred]
			m += 1

	return inds_pred, s_cc, s_cp, s_pp

# plot settings
lw = 1.5
mp.rc('font', family = 'serif')
mp.rcParams['text.latex.preamble'] = [r'\boldmath']
mp.rcParams['axes.linewidth'] = lw
mp.rcParams['lines.linewidth'] = lw
cm = mpcm.get_cmap('plasma')

# setup
constrain = False
precompress = False
inpaint = False
n_spectra = 29502
n_classes = 1
n_samples = 10000 # 1000
diagnose = False
datafile = 'data/redclump_{:d}_alpha_nonorm.h5' # filename or None
window = 'data/centers_subset2_ce_nd.txt' # filename or None
eval_thresh = 1.0e-4

# build up output filename
if datafile is None:
	io_base = 'simple_test_'
else:
	io_base = 'apogee_'
	if window is not None:
		window_file = window.split('/')[1]
		io_base += window_file.split('.')[0] + '_'
io_base += '{:d}_spc_'.format(n_spectra)
if n_classes > 1:
	io_base += '{:d}_cls_'.format(n_classes)
if precompress:
	if inpaint:
		io_base += 'inpaint_pca_'
	else:
		io_base += 'pca_'


# retrieve wavelengths
n_to_load = 1
n_file = 1
wl, full_data = read_spectra(n_to_load, \
							 datafile.format(n_file), \
							 return_wl=True)

# read in window definitions. file contains elements with 
# positions of features within three wavelength ranges. take 
# windows of +/- 2.5 Angstroms about each line center; centers
# of 999 Angstroms should be ignored
wdata = np.genfromtxt(window, dtype=None, \
					  skip_header=1)
centers, elements = [], []
for i in range(len(wdata)):
	#for j in range(3):
	for j in range(2):
		center = wdata[i][j + 1]
		if center != 999.0:
			centers.append(center)
			elements.append(wdata[i][0])
centers, elements = (list(t) for t in \
					 zip(*sorted(zip(centers, elements))))
windices = np.full(len(wl), False, dtype=bool)
print 'selecting wavelengths within 2.5 Angstroms of:'
wlabels = [elements[0]]
for i in range(len(centers)):
	windices = np.logical_or(windices, (wl >= centers[i] - 2.5) & \
									   (wl <= centers[i] + 2.5))
	msg = '{0:d}: {1:.2f} A (' + elements[i] + ')'
	print msg.format(i, centers[i])
	if i > 0:
		if np.abs(centers[i-1] - centers[i]) < 5.0:
			wlabels[-1] += '/' + elements[i]
		else:
			wlabels.append(elements[i])

# determine the selected-data indices where the breaks are
n_bins = np.sum(windices)
dwindices = np.append(0, windices[1:].astype(int) - \
						 windices[:-1].astype(int))
wendows = [x[0] for x in np.argwhere(dwindices < 0)]
wbegindows = [x[0] for x in np.argwhere(dwindices > 0)]
wl_begin = wl[wbegindows]
wl_end = wl[wendows]
n_windows = len(wendows)
wendices = []
for i in range(n_windows):
	wendices.append(np.sum(windices[0: wendows[i]]))
n_in_bin = np.append(wendices[0], np.diff(wendices))
wl = np.arange(n_bins)

# retrieve samples
with h5py.File(io_base + 'samples.h5', 'r') as f:
	mean_samples = f['mean'][:]
	cov_samples = f['covariance'][:]
	n_bins, n_classes, n_samples = mean_samples.shape
	if n_classes > 1:
		class_probs_samples = f['class_probs'][:]
	n_warmup = n_samples / 4

# reproject compressed mean and covariance samples back onto original
# spectral bins
if precompress:

	if rank == 0:
		print 'inflating samples'
	mean_samples = np.einsum('ji,jkl', proj_pc, mean_samples)
	tmp = np.einsum('ji,jlmn', proj_pc, cov_samples)
	cov_samples = np.einsum('iklm,kj', tmp, proj_pc)
	#cov_samples = np.einsum('ki,klmn,lj', proj_pc, cov_samples, proj_pc)
	n_bins = n_bins_in

# summarize results
mp_mean = np.zeros((n_bins, n_classes))
sdp_mean = np.zeros((n_bins, n_classes))
mp_cov = np.zeros((n_bins, n_bins, n_classes))
mp_cor = np.zeros((n_bins, n_bins, n_classes))
for k in range(n_classes):
	mp_mean[:, k] = np.mean(mean_samples[:, k, n_warmup:], -1)
	sdp_mean[:, k] = np.std(mean_samples[:, k, n_warmup:], -1)
	mp_cov[:, :, k] = np.mean(cov_samples[:, :, k, n_warmup:], -1)
	for i in range(n_bins):
		for j in range(n_bins):
			mp_cor[i, j, k] = \
				np.mean(cov_samples[i, j, k, n_warmup:] / \
						np.sqrt(cov_samples[i, i, k, n_warmup:] * \
								cov_samples[j, j, k, n_warmup:]))
if n_classes > 1:
	mp_class_probs = np.mean(class_probs_samples, 1)
	if datafile is None and rank == 0:

		# identify the closest match between true and MP classes
		# this scales factorially with the number of classes, fyi
		# might not want to do it.
		n_perms = int(spesh.factorial(n_classes))
		chisq = np.zeros(n_perms)
		perms = list(it.permutations(range(n_classes)))
		mp_cov_inv = np.zeros((n_bins, n_bins, n_classes))
		for k in range(n_classes):
			mp_cov_inv[:, :, k] = npl.inv(mp_cov[:, :, k])
		for p in range(n_perms):
			for k in range(n_classes):
				m = perms[p][k]
				res = mp_mean[:, m] - mean[:, k]
				chisq[p] += np.dot(res, np.dot(mp_cov_inv[:, :, k], res))
		mp_perm = perms[np.argmin(chisq)]
		print 'best permutation is', mp_perm
		print 'perms: ', perms
		print 'chi squares: ', chisq

else:
	if datafile is None:
		mp_perm = [0]

# conditional variance plots: which features best predict others?
# this might produce the most gigantic plots ever...
n_ext = 10
fig, axes = mp.subplots(n_windows, n_classes, \
						figsize=(8 * n_classes, 3 * n_windows), \
						sharex=True)
fig_e, axes_e = mp.subplots(n_ext / 2, n_classes, \
							figsize=(8 * n_classes, 3 * n_ext / 2), \
							sharex=True)
fig_d, axes_d = mp.subplots(n_classes, 1, \
							figsize=(8, 5 * n_classes), \
							sharex=True)
if n_classes == 1:
	axes = axis_to_axes(axes, True)
	axes_e = axis_to_axes(axes_e, True)
	axes_d = axis_to_axes(axes_d)
for k in range(n_classes):

	inf_gain = np.zeros(n_windows ** 2)
	stddev = np.zeros((n_bins, n_bins))
	for i in range(n_windows):

		# split covariance matrix. calculate and plot covariance of 
		# conditional distribution
		inds_o, s_ii, s_io, s_oo = split_covariance(i, mp_cov[:, :, k], \
													wendices)
		n_o = np.sum(inds_o)
		s_ii_inv = npl.inv(s_ii)
		cond_cov = s_oo - np.dot(s_io.T, np.dot(s_ii_inv, s_io))
		stddev[i, inds_o] = np.sqrt(np.diag(cond_cov))
		axes[i, k].plot(wl[inds_o], np.sqrt(np.diag(cond_cov)))
		axes[i, k].set_ylabel(r'$\sigma$')

		# also plot information gain for each window from each
		# other window
		others = [w for w in range(n_windows) if w != i]
		for w in others:

			# find indices
			inds_w = np.full(n_o, False, dtype=bool)
			if i < w:
				inds_w_min = np.sum(n_in_bin[0: w]) - n_in_bin[i]
			else:
				inds_w_min = np.sum(n_in_bin[0: w])
			inds_w_max = inds_w_min + n_in_bin[w]
			inds_w[inds_w_min: inds_w_max] = True
			n_w = n_in_bin[w]

			# construct submatrices
			s_ww = np.zeros((n_w, n_w))
			cond_cov_ww = np.zeros((n_w, n_w))
			n = 0
			for j in range(n_o):
				if inds_w[j]:
					s_ww[n, :] = s_oo[j, inds_w]
					cond_cov_ww[n, :] = cond_cov[j, inds_w]
					n += 1

			# calculate information gain
			inf_gain[w * n_windows + i] = \
				np.log(npl.det(cond_cov_ww)) - \
				np.log(npl.det(s_ww))

	# plot information gains
	ylim = [ax.get_ylim() for ax in axes[:, k]]
	ylim = [np.min(ylim[:][0]), np.max(ylim[:][1])]
	for i in range(n_windows):
		axes[i, k].set_ylim(ylim)
		for n in range(n_windows):
			axes[i, k].axvline(wendices[n], color='k', lw=0.5, ls=':')
			axes[i, k].text(wendices[n], axes[i, k].get_ylim()[0], \
							wlabels[n], fontsize=8, ha='right', \
							va='bottom')
	ind_plot = np.arange(n_windows ** 2)
	axes_d[k].scatter(ind_plot, inf_gain, marker='+')
	i_mig = np.argsort(inf_gain)
	ind_ext = (inf_gain < inf_gain[i_mig[n_ext]])
	axes_d[k].scatter(ind_plot[ind_ext], \
					  inf_gain[ind_ext], marker='+', \
					  color='r')
	for i in range(n_windows):
		axes_d[k].axvline(i * n_windows, color='k', lw=0.5)
	xticks = [(i + 0.5) * n_windows for i in range(n_windows)]
	xticklabels = ['{:d}'.format(i) for i in range(n_windows)]
	axes_d[k].set_xticks(xticks)
	axes_d[k].set_xticklabels(wlabels, fontdict={'fontsize':11})
	axes_d[k].tick_params(axis='x', which='both', length=0)
	axes_d[k].set_xlim(0, n_windows ** 2)
	y_min = np.min(inf_gain)
	y_max = np.max(inf_gain[inf_gain < 0])
	axes_d[k].set_ylim(y_min - np.abs(y_max) * 0.1, y_max)
	axes_d[k].set_ylabel(r'$\log|C_{ii|j}| - \log|C_{ii}|$')

	# plot std devs of pairs with most information gain
	test = np.arange(n_windows) * n_windows
	for n in range(n_ext / 2):
		cols = ['b', 'g']
		for m in range(2):
			j = np.searchsorted(test, i_mig[n * 2 + m], side='right') - 1
			i = i_mig[n * 2 + m] - j * n_windows
			label = r'$\sqrt{{\rm diag}\left(C_{\rm X|' + \
					wlabels[i] + r'}\right)}$'
			ind_0 = np.where(stddev[i, :] == 0)[0]
			axes_e[n, k].plot(wl[0: ind_0[0]], stddev[i, 0: ind_0[0]], \
							  color=cols[m], label=label)
			axes_e[n, k].plot(wl[ind_0[-1] + 1:], \
							  stddev[i, ind_0[-1] + 1:], \
							  color=cols[m])
	ylim = [ax.get_ylim() for ax in axes_e[:, k]]
	ylim = [np.min(ylim[:][0]), np.max(ylim[:][1])]
	for i in range(n_ext / 2):
		axes_e[i, k].set_ylim(ylim)
		axes_e[i, k].set_ylabel(r'$\sigma$')
		yticks = axes_e[i, k].yaxis.get_major_ticks()
		yticks[0].label1.set_visible(False)
		axes_e[i, k].legend(loc='upper right', fontsize=12)
		axes_e[i, k].tick_params(axis='x', which='both', length=0)
		jj = np.searchsorted(test, i_mig[i * 2], side='right') - 1
		ii = i_mig[i * 2] - jj * n_windows
		if ii > jj:
			if jj > 0:
				axes_e[i, k].axvspan(0, wendices[jj - 1], \
									 alpha=0.2, color='gray')
			if ii != jj + 1:
				axes_e[i, k].axvspan(wendices[jj], \
									 wendices[ii - 1], \
									 alpha=0.2, color='gray')
			axes_e[i, k].axvspan(wendices[ii], wl[-1], \
								 alpha=0.2, color='gray')
		else:
			if ii > 0:
				axes_e[i, k].axvspan(0, wendices[ii - 1], \
									 alpha=0.2, color='gray')
			if jj != ii + 1:
				axes_e[i, k].axvspan(wendices[ii], \
									 wendices[jj - 1], \
									 alpha=0.2, color='gray')
			axes_e[i, k].axvspan(wendices[jj], wl[-1], \
								 alpha=0.2, color='gray')
		for n in range(n_windows):
			x_text = n_in_bin[n] / 2.0
			if n > 0:
				x_text += wendices[n - 1]
			axes_e[i, k].text(x_text, ylim[0], \
							  wlabels[n], fontsize=8, \
							  ha='center', va='bottom')
			axes_e[i, k].axvline(wendices[n], color='k', \
								 lw=0.5, ls=':')

# finish and save plots
axes[-1, k].set_xlabel(r'${\rm index},\,i$', fontsize=16)
axes[-1, k].set_xlim(wl[0], wl[-1])
axes_e[-1, k].set_xlabel(r'${\rm index},\,i$', fontsize=16)
axes_e[-1, k].set_xlim(wl[0], wl[-1])
fig.subplots_adjust(hspace=0)
fig.savefig(io_base + 'conditional_stddevs.pdf', bbox_inches='tight')
mp.close(fig)
fig_d.subplots_adjust(hspace=0)
fig_d.savefig(io_base + 'conditional_inf_gain.pdf', bbox_inches='tight')
mp.close(fig_d)
fig_e.subplots_adjust(hspace=0)
fig_e.savefig(io_base + 'most_correlated_stddevs.pdf', bbox_inches='tight')
mp.close(fig_e)

# additional plots for target elements
tgts = ['Ce', 'Nd']
for tgt in tgts:
	tgt_windows = [ind for ind, label in enumerate(wlabels) \
				   if label == tgt]
	n_tgt_windows = len(tgt_windows)
	if n_tgt_windows > 0:

		# plot predicted std devs (and covs?) for Cerium windows
		# given all other data
		fig, axes = mp.subplots(n_classes, n_tgt_windows, \
								figsize=(8 * n_tgt_windows, \
										 5 * n_classes))
		fig_sd, axes_sd = mp.subplots(n_classes, n_tgt_windows, \
									  figsize=(8 * n_tgt_windows, \
											   5 * n_classes))
		if n_classes == 1:
			axes = axis_to_axes(axes, True)
			axes_sd = axis_to_axes(axes_sd, True)
		for k in range(n_classes):

			# loop over Cerium windows
			for i in range(n_tgt_windows):

				# iteratively find most predictive other window
				tot_inf_gain = []
				most_pred_w = []
				stddev = []
				for j in range(n_windows - 1):

					# others is a list of the windows that are
					# 1) not being predicted and
					# 2) not already in the best predictors list
					inf_gain = np.zeros(n_windows)
					stddevs = np.zeros((n_in_bin[tgt_windows[i]], \
										n_windows))
					others = np.setdiff1d(range(n_windows), most_pred_w)
					others = np.setdiff1d(others, [tgt_windows[i]])

					# want covariance in target Cerium window conditioned 
					# on each "others" window *and* all most_pred_w. 
					for w in others:

						# split covariance matrix accordingly and calculate 
						# conditional covariance
						i_cond = [w] + most_pred_w
						inds_o, s_ii, s_io, s_oo = \
							split_covariance(i_cond, mp_cov[:, :, k], \
											 wendices, i_pred=tgt_windows[i])
						s_ii_inv = npl.inv(s_ii)
						cond_cov = s_oo - np.dot(s_io.T, np.dot(s_ii_inv, s_io))
						
						# calculate information gain and prediction std dev
						inf_gain[w] = \
							np.log(npl.det(cond_cov)) - \
							np.log(npl.det(s_oo))
						stddevs[:, w] = np.sqrt(np.diag(cond_cov))

					# find most predictive window
					i_mig = np.argsort(inf_gain)
					tot_inf_gain.append(inf_gain[i_mig[0]])
					most_pred_w.append(i_mig[0])
					stddev.append(stddevs[:, i_mig[0]])

				# plot information gains!
				cm = mpcm.get_cmap('plasma')
				cols = [cm(x) for x in np.linspace(0.1, 0.8, n_windows - 1)]
				if n_tgt_windows == 1:
					ax = axes[k]
				else:
					ax = axes[i, k]
				ax.plot(tot_inf_gain)
				for j in range(n_windows - 2):
					ax.plot([j, j + 1], tot_inf_gain[j: j + 2], color=cols[j])
					ax.axvline(j, ls='--', color='grey', zorder=0)
				ax.set_xlabel('element')
				ax.set_ylabel(r'$\log|C_{ii|j}| - \log|C_{ii}|$')
				xticklabels = [wlabels[mpw] for mpw in most_pred_w]
				ax.set_xticks(np.arange(n_windows - 1))
				ax.set_xticklabels(xticklabels, rotation=90)
				ax.set_xlim(0, n_windows - 2)

				# plot std devs!
				if n_tgt_windows == 1:
					ax = axes_sd[k]
				else:
					ax = axes_sd[i, k]
				x_bar = np.rint(0.5 * (wl_begin[tgt_windows[i]] + \
										wl_end[tgt_windows[i]]))
				x = np.linspace(wl_begin[tgt_windows[i]], \
								wl_end[tgt_windows[i]], \
								len(stddev[j])) - x_bar
				for j in range(len(stddev)):
					ax.plot(x, stddev[j], color=cols[j])
				ax.set_xlabel(r'$\lambda-' + '{:5d}'.format(int(x_bar)) + \
							  r'\,[{\rm Angstroms}]$')
				ax.set_ylabel(r'$\sigma$')
				ax.set_xlim(wl_begin[tgt_windows[i]] - x_bar, \
							wl_end[tgt_windows[i]] - x_bar)

		# finish off plots
		fig.subplots_adjust(hspace=0)
		fig.savefig(io_base + tgt.lower() + '_inf_gain.pdf', \
					bbox_inches='tight')
		mp.close(fig)
		fig_sd.subplots_adjust(hspace=0)
		fig_sd.savefig(io_base + tgt.lower() + '_conditional_stddevs.pdf', \
					   bbox_inches='tight')
		mp.close(fig_sd)

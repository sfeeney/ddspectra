import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import scipy.stats as sps
import scipy.linalg.lapack as spl
import h5py
import os
if 'DISPLAY' not in os.environ.keys():
	import matplotlib
	matplotlib.use('Agg')
import matplotlib.pyplot as mp
import matplotlib.cm as mpcm
import matplotlib.axes as mpa
#import corner as co

def symmetrize(m):
	return (m + m.T) / 2.0

def allocate_jobs(n_jobs, n_procs=1, rank=0):
	n_j_allocated = 0
	for i in range(n_procs):
		n_j_remain = n_jobs - n_j_allocated
		n_p_remain = n_procs - i
		n_j_to_allocate = n_j_remain / n_p_remain
		if rank == i:
			return range(n_j_allocated, \
						 n_j_allocated + n_j_to_allocate)
		n_j_allocated += n_j_to_allocate

def complete_array(target_distrib, use_mpi=False):
	if use_mpi:
		target = np.zeros(target_distrib.shape, \
						  dtype=target_distrib.dtype)
		#mpi.COMM_WORLD.Reduce(target_distrib, target, op=mpi.SUM, \
		#					  root=0)
		mpi.COMM_WORLD.Allreduce(target_distrib, target, op=mpi.SUM)
	else:
		target = target_distrib
	return target

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

# @TODO LIST
#  - impose low rank on covariance

# plot settings
lw = 1.5
mp.rc('font', family = 'serif')
mp.rcParams['text.latex.preamble'] = [r'\boldmath']
mp.rcParams['axes.linewidth'] = lw
mp.rcParams['lines.linewidth'] = lw
cm = mpcm.get_cmap('plasma')

# setup
use_mpi = True
constrain = True
no_s_inv = False
sample = False
precompress = True
n_bins = 7 # 50
n_spectra = 29502
n_classes = 1
n_samples = 500 # 1000
n_warmup = n_samples / 4
n_gp_reals = 50
jeffreys_prior = 1
diagnose = False
datafile = 'data/redclump_{:d}_alpha_nonorm.h5' # filename or None
window = True
inf_noise = 1.0e5
reg_noise = 1.0e-6
eval_thresh = 1.0e-2

# set up identical within-chain MPI processes
if use_mpi:
	import mpi4py.MPI as mpi
	n_procs = mpi.COMM_WORLD.Get_size()
	rank = mpi.COMM_WORLD.Get_rank()
else:
	n_procs = 1
	rank = 0
if constrain:
	seed = 231014 + rank
else:
	seed = npr.randint(231014, 221216, 1)[0]
npr.seed(seed)

# simulate or read input data
if datafile is None:

	# initialize
	data = np.zeros((n_spectra, n_bins))
	mask = np.zeros((n_spectra, n_bins))
	if no_s_inv:
		cov_noise = np.zeros((n_spectra, n_bins))
	else:
		inv_cov_noise = np.zeros((n_bins, n_bins))
	wl = np.zeros(n_bins)
	mean = np.zeros((n_bins, n_classes))
	cov = np.zeros((n_bins, n_bins, n_classes))
	cor = np.zeros((n_bins, n_bins, n_classes))
	class_ids = np.zeros(n_spectra, dtype=int)
	if rank == 0:

		# additional setup: use BW setup or test input matching prior
		test_match = False
		if test_match:

			# generate a covariance matrix: HOW? not inv wishart prior, 
			# but something about determinant. so how do we guarantee 
			# that?
			norm_vars = npr.randn(n_bins)
			in_mat = np.zeros((n_bins, n_bins))
			for i in range(n_bins):
				in_mat[:, i] = norm_vars
			u, s, v = npl.svd(in_mat)
			evex = u
			evals = np.linspace(1.0, n_bins, n_bins) ** -5
			cov = np.dot(np.dot(evex, np.diag(evals)), evex.T)
			evals, evex = npl.eigh(cov)

			# generate a mean vector
			mean = npr.uniform(0.5, 1.5, n_bins)

		else:

			# loop over classes
			for k in range(n_classes):

				# generate a covariance matrix
				norm_vars = npr.randn(n_bins)
				in_mat = np.zeros((n_bins, n_bins))
				for i in range(n_bins):
					in_mat[:, i] = norm_vars
				u, s, v = npl.svd(in_mat)
				evex = u
				evals = np.linspace(1.0, n_bins, n_bins) ** -5
				cov[:, :, k] = np.dot(np.dot(evex, np.diag(evals)), evex.T)
				for i in range(n_bins):
					for j in range(n_bins):
						cor[i, j, k] = cov[i, j, k] / \
									   np.sqrt(cov[i, i, k] * \
											   cov[j, j, k])
				
				# generate a mean vector
				#mean[:, k] = k + 1.0 + 4.0 * (npr.rand(n_bins) - 0.5)
				mean[:, k] = 1.0 + 4.0 * (npr.rand(n_bins) - 0.5)

		for k in range(n_classes):
			print 'input covmat {:d} condition number '.format(k), \
				  npl.cond(cov[:, :, k])

		# optionally plot test inputs
		if diagnose:
			for k in range(n_classes):
				mp.imshow(cov[:, :, k], interpolation='nearest')
				mp.show()
				mp.plot(mean[:, k])
				mp.show()

		# assign class membership and generate signals from this model
		class_ids = npr.choice(n_classes, n_spectra)
		spectra_true = np.zeros((n_spectra, n_bins))
		for k in range(n_classes):
			in_class_k = (class_ids == k)
			spectra_true[in_class_k, :] = \
				npr.multivariate_normal(mean[:, k], cov[:, :, k], \
										np.sum(in_class_k))
		if diagnose:
			for i in range(n_spectra):
				mp.plot(spectra_true[i, :], 'k-', alpha=0.3)
			mp.show()

		# generate noise and mask
		var_noise = npr.uniform(0.0, 0.2, n_bins)
		cov_noise = np.diag(var_noise)
		inv_cov_noise = npl.inv(cov_noise)	  # inverseNoiseCov
		noise = npr.multivariate_normal(np.zeros(n_bins), cov_noise, n_spectra)
		mask = np.reshape(npr.choice(2, n_bins * n_spectra, p=[0.14, 0.86]), \
						  spectra_true.shape).astype(float)
		if no_s_inv:
			# this is weird, but preserves random seeds, so useful. 
			# adjust the above draws to have "infinite" noise where 
			# masked
			cov_noise = np.zeros((n_spectra, n_bins))
			for i in range(n_spectra):
				cov_noise[i, :] = var_noise
				masked = (mask[i, :] == 0)
				cov_noise[i, masked] = inf_noise
				noise[i, masked] *= inf_noise / var_noise[masked]
		if diagnose:
			for i in range(n_spectra):
				mp.plot(noise[i, :], 'k-', alpha=0.3)
			mp.show()

		# generate data
		data = mask * (spectra_true + noise)
		wl = np.arange(n_bins).astype(float)
		if diagnose:
			print float(np.sum(mask==1.0)) / float(np.sum(mask==0.0)), 0.86/0.14
			mp.imshow(data, interpolation='nearest')
			mp.show()

	# ensure all processes have same data
	if use_mpi:
		mpi.COMM_WORLD.Bcast(data, root=0)
		mpi.COMM_WORLD.Bcast(mask, root=0)
		if no_s_inv:
			mpi.COMM_WORLD.Bcast(cov_noise, root=0)
		else:
			mpi.COMM_WORLD.Bcast(inv_cov_noise, root=0)
		mpi.COMM_WORLD.Bcast(wl, root=0)
		mpi.COMM_WORLD.Bcast(mean, root=0)
		mpi.COMM_WORLD.Bcast(cov, root=0)
		mpi.COMM_WORLD.Bcast(cor, root=0)
		mpi.COMM_WORLD.Bcast(class_ids, root=0)

else:

	# read in data
	n_to_load = n_spectra
	n_file = 1
	full_data = None
	while n_to_load > 0:
		datafile_n = datafile.format(n_file)
		try:
			f = h5py.File(datafile_n, 'r')
		except:
			msg = 'ERROR: input file ' + datafile_n + ' not found'
			if use_mpi:
				if rank == 0:
					print msg
				mpi.COMM_WORLD.Abort()
			else:
				exit(msg)
		file_data = f['dataset_1'][:]
		n_in_file = file_data.shape[1]
		to_load = np.arange(n_in_file) < n_to_load
		if full_data is None:
			wl = file_data[:, 0, 0]
			full_data = file_data[:, to_load, 1:]
		else:
			full_data = np.append(full_data, \
								  file_data[:, to_load, 1:], 1)
		if rank == 0:
			print 'loaded ' + \
				  '{:d}/{:d}'.format(np.sum(to_load), n_spectra) + \
				  ' spectra from ' + datafile_n
		n_to_load -= n_in_file
		n_file += 1

	# construct data vector and noise covariance: mask handled by 
	# noise variance
	if window:

		# read in window definitions. file contains elements with 
		# positions of features within three wavelength ranges. take 
		# windows of +/- 2.5 Angstroms about each line center; centers
		# of 999 Angstroms should be ignored
		wdata = np.genfromtxt('data/centers_subset2.txt', dtype=None, \
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
		if rank == 0:
			print 'selecting wavelengths within 2.5 Angstroms of:'
		wlabels = [elements[0]]
		for i in range(len(centers)):
			windices = np.logical_or(windices, (wl >= centers[i] - 2.5) & \
											   (wl <= centers[i] + 2.5))
			if rank == 0:
				msg = '{0:d}: {1:.2f} A (' + elements[i] + ')'
				print msg.format(i, centers[i])
			if i > 0:
				if np.abs(centers[i-1] - centers[i]) < 5.0:
					wlabels[-1] += '/' + elements[i]
				else:
					wlabels.append(elements[i-1])

		# select data
		n_bins = np.sum(windices)
		wl = np.arange(n_bins)
		data = full_data[windices, 0: n_spectra, 0].T
		var_noise = full_data[windices, 0: n_spectra, 1].T ** 2
		inv_cov_noise = np.zeros((n_spectra, n_bins, n_bins))
		for i in range(n_spectra):
			inv_cov_noise[i, :, :] = np.diag(1.0 / var_noise[i, :])

		# determine the selected-data indices where the breaks are
		dwindices = np.append(0, windices[1:].astype(int) - \
								 windices[:-1].astype(int))
		wendows = [x[0] for x in np.argwhere(dwindices < 0)]
		n_windows = len(wendows)
		wendices = []
		for i in range(n_windows):
			wendices.append(np.sum(windices[0: wendows[i]]))
		n_in_bin = np.append(wendices[0], np.diff(wendices))

	else:

		i_min = 1770 # 1785
		if rank == 0:
			msg = 'selecting wavelengths in range {0:.2f}-{1:.2f} Angstroms'
			print msg.format(wl[i_min], wl[i_min + n_bins])
		wl = wl[i_min: i_min + n_bins] - 15100.0
		data = full_data[i_min: i_min + n_bins, 0: n_spectra, 0].T
		var_noise = full_data[i_min: i_min + n_bins, 0: n_spectra, 1].T ** 2
		inv_cov_noise = np.zeros((n_spectra, n_bins, n_bins))
		for i in range(n_spectra):
			inv_cov_noise[i, :, :] = np.diag(1.0 / var_noise[i, :])

	# plot if desired
	if diagnose and rank == 0:
		n_to_plot = 20 # n_spectra
		cols = [cm(x) for x in np.linspace(0.1, 0.9, n_to_plot)]
		fig, axes = mp.subplots(1, 2, figsize=(10, 5))
		for i in range(n_to_plot):
			axes[0].plot(wl, data[i, :], color=cols[i], alpha=0.5)
			axes[1].semilogy(wl, np.sqrt(var_noise[i, :]), \
							 color=cols[i], alpha=0.5)
		for ax in axes:
			ax.set_xlim(wl[0], wl[-1])
			ax.set_xlabel(r'$\lambda-15100\,[{\rm Angstroms}]$')
			mp.setp(ax.get_xticklabels(), rotation=45)
		axes[0].set_ylabel(r'${\rm flux}$')
		axes[1].set_ylabel(r'$\sigma_{\rm flux}$')
		if window:
			for i in range(n_windows):
				axes[0].axvline(wendices[i], color='k', lw=0.5)
				axes[1].axvline(wendices[i], color='k', lw=0.5)
		mp.subplots_adjust(bottom=0.15)
		mp.savefig('simple_test_apogee_inputs.pdf', \
				   bbox_inches='tight')
		mp.show()

# perform a PCA of the input data
if precompress:
	import sklearn.decomposition as skd
	pca = skd.PCA()
	pca.fit(data)
	d_evals = pca.explained_variance_[::-1]
	d_evex = pca.components_[::-1, :]
	ind_pc_sig = d_evals > \
				 np.max(d_evals) * eval_thresh
	n_pc_sig = np.sum(ind_pc_sig)

# initial conditions for sampler
class_id_sample = np.zeros(n_spectra, dtype=int)
if n_classes > 1:
	if rank == 0:
		class_id_sample = npr.choice(n_classes, n_spectra)
	if use_mpi:
		mpi.COMM_WORLD.Bcast(class_id_sample, root=0)
mean_sample = np.zeros((n_bins, n_classes))
cov_sample = np.zeros((n_bins, n_bins, n_classes))
spectra_samples = np.zeros((n_spectra, n_bins))
for k in range(n_classes):
	in_class_k = (class_id_sample == k)
	mean_sample[:, k] = np.mean(data[in_class_k, :], 0)
	cov_sample[:, :, k] = np.cov(data[in_class_k, :], rowvar=False)
for j in range(n_spectra):
	#spectra_samples[j, :] = mean_sample[:, class_id_sample[j]]
	spectra_samples[j, :] = data[j, :]
mean_samples = np.zeros((n_bins, n_classes, n_samples))
cov_samples = np.zeros((n_bins, n_bins, n_classes, n_samples))
conds = np.zeros((n_classes, n_samples))

# Gibbs sample!
if sample:

	job_list = allocate_jobs(n_spectra, n_procs, rank)
	class_job_list = allocate_jobs(n_classes, n_procs, rank)
	d_sample = n_samples / 10
	for i in range(n_samples):

		# report progress
		if np.mod(i, d_sample) == 0 and rank == 0:
			print i, '/', n_samples

		# invert current sample covariance and obtain determinant. 
		# use lapack directly for this to avoid two N^3 operations
		if not no_s_inv:
			inv_cov_sample = np.zeros((n_bins, n_bins, n_classes))
			sqrt_cov_det = np.zeros(n_classes)
			for k in range(n_classes):
				#inv_cov_sample[:, :, k] = npl.inv(cov_sample[:, :, k])
				chol_k = spl.dpotrf(cov_sample[:, :, k])[0]
				inv_cov_sample[:, :, k] = spl.dpotri(chol_k)[0]
				for j in range(n_bins):
					inv_cov_sample[j:, j, k] = inv_cov_sample[j, j:, k]
				sqrt_cov_det[k] = np.product(np.diagonal(chol_k))

		# reassign classes
		class_id_sample = np.zeros(n_spectra, dtype=int)
		if n_classes > 1:
			for j in job_list:
				class_probs = np.zeros(n_classes)
				for k in range(n_classes):
					delta = spectra_samples[j, :] - mean_sample[:, k]
					chisq = np.dot(delta, \
								   np.dot(inv_cov_sample[:, :, k], \
								   		  delta))
					class_probs[k] = np.exp(-0.5 * chisq) / \
									 sqrt_cov_det[k]
				class_probs /= np.sum(class_probs)
				class_id_sample[j] = npr.choice(n_classes, \
												p=class_probs)
			class_id_sample = complete_array(class_id_sample, use_mpi)

		# calculate WF for each spectrum and use to draw true spectra
		spectra_samples = np.zeros(data.shape)
		for j in job_list:

			# class id
			k = class_id_sample[j]

			# avoid S^-1?
			if no_s_inv:

				# check if have spectrum-dependent noise
				# @TODO: update to support multiple classes
				if len(cov_noise.shape) == 3:
					cov_noise_j = cov_noise[j, :, :]
				else:
					cov_noise_j = np.diag(cov_noise[j, :]) # noisy mask included
				s_n_inv = npl.inv(cov_sample + cov_noise_j) 
				mean_wf = np.dot(cov_noise_j, \
								 np.dot(s_n_inv, mean_sample)) + \
						  np.dot(cov_sample, \
								 np.dot(s_n_inv, data[j, :]))
				cov_wf = np.dot(cov_noise_j, np.dot(s_n_inv, cov_sample))
				cov_wf = symmetrize(cov_wf)

			else:

				# check if have spectrum-dependent noise
				# @TODO: can i speed up this inversion given i already 
				#        have the cholesky decomp of S?
				if len(inv_cov_noise.shape) == 3:
					inv_cov_noise_j = inv_cov_noise[j, :, :]
				else:
					inv_cov_noise_j = np.outer(mask[j, :], mask[j, :]) * \
									  inv_cov_noise
				cov_wf = npl.inv(inv_cov_sample[:, :, k] + \
								 inv_cov_noise_j)
				cov_wf = symmetrize(cov_wf)
				mean_wf = np.dot(cov_wf, \
								 np.dot(inv_cov_sample[:, :, k], \
								 		mean_sample[:, k]) + \
								 np.dot(inv_cov_noise_j, data[j, :]))
				
			spectra_samples[j, :] = npr.multivariate_normal(mean_wf, \
															cov_wf, 1)
		spectra_samples = complete_array(spectra_samples, use_mpi)

		# only class parameters require sampling; can do one class per 
		# mpi process. first sample means
		mean_sample = np.zeros((n_bins, n_classes))
		for k in class_job_list:

			# class members
			in_class_k = (class_ids == k)
			n_spectra_k = np.sum(in_class_k)

			# sample signal mean
			#mean_sample[:, k] = \
			#	npr.multivariate_normal(np.mean(spectra_samples[:], 0), \
			#							cov_sample / n_spectra, 1)[0, :]
			mean_mean_k = np.mean(spectra_samples[in_class_k, :], 0)
			mean_sample[:, k] = \
				npr.multivariate_normal(mean_mean_k, \
										cov_sample[:, :, k] / \
										n_spectra_k, 1)[0, :]

		mean_sample = complete_array(mean_sample, use_mpi)

		# now sample covariances
		cov_sample = np.zeros((n_bins, n_bins, n_classes))
		for k in class_job_list:

			# class members
			in_class_k = (class_ids == k)
			n_spectra_k = np.sum(in_class_k)

			# sample signal covariance matrix
			# NB: scipy.stats uses numpy.random seed, which i've already set
			n_dof = n_spectra_k - (1 - jeffreys_prior) * (n_bins + 1)
			sigma = np.zeros((n_bins, n_bins))
			for j in range(n_spectra):
				if in_class_k[j]:
					delta = spectra_samples[j, :] - mean_sample[:, k]
					sigma += np.outer(delta, delta)
			cov_sample[:, :, k] = sps.invwishart.rvs(n_dof, sigma, 1) + \
								  np.diag(np.ones(n_bins) * reg_noise)

		cov_sample = complete_array(cov_sample, use_mpi)
		
		# store samples (marginalize over true spectra)
		mean_samples[:, :, i] = mean_sample
		cov_samples[:, :, :, i] = cov_sample
		for k in range(n_classes):
			conds[k, i] = npl.cond(cov_sample[:, :, k])

		# report if desired
		if diagnose and rank == 0:
			print 'sampled cov mat condition numbers:', conds[:, i]

	# store samples on disk
	if rank == 0:
		with h5py.File('simple_test_samples.h5', 'w') as f:
			f.create_dataset('mean', data=mean_samples)
			f.create_dataset('covariance', data=cov_samples)

else:

	# retrieve samples
	if rank == 0:
		with h5py.File('simple_test_samples.h5', 'r') as f:
			mean_samples = f['mean'][:]
			cov_samples = f['covariance'][:]
			n_bins, n_classes, n_samples = mean_samples.shape
			n_warmup = n_samples / 4

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

# plots
if rank == 0:

	# selection of trace plots
	fig, axes = mp.subplots(3, 1, figsize=(8, 5), sharex=True)
	for k in range(n_classes):
		axes[0].plot(mean_samples[0, k, :])
		axes[1].plot(mean_samples[n_bins / 2, k, :])
		axes[2].plot(mean_samples[-1, k, :])
	axes[2].set_xlabel('sample')
	axes[1].set_ylabel('mean')
	fig.subplots_adjust(hspace=0, wspace=0)
	mp.savefig('simple_test_trace.pdf', bbox_inches='tight')
	mp.close()

	# compare means with reference to noise and posterior standard 
	# deviations
	fig, axes = mp.subplots(n_classes, 1, \
							figsize=(8, 5 * n_classes), \
							sharex=True)
	if n_classes == 1:
		axes = axis_to_axes(axes)
	for k in range(n_classes):
		if datafile is None:
			res = mp_mean[:, k] - mean[:, k]
			label = 'residual'
		else:
			res = mp_mean[:, k]
			label = 'posterior mean'
		axes[k].fill_between(wl, res - sdp_mean[:, k], \
							 res + sdp_mean[:, k], \
							 color='LightGrey', \
							 label=r'posterior $\sigma$')
		axes[k].plot(wl, res, 'r', label=label)
		if datafile is not None:
			if window:
				for i in range(n_windows):
					axes[k].axvline(wendices[i], color='k', lw=0.5, \
									ls=':')
					axes[k].text(wendices[i], mp.gca().get_ylim()[0], \
								 wlabels[i], fontsize=8, ha='right', \
								 va='bottom')
			axes[k].set_ylabel(r'$\mu^{\rm post}$', fontsize=14)
		else:
			axes[k].plot([wl[0], wl[-1]], [0.0, 0.0], 'k--')
			axes[k].set_ylabel(r'$\mu_i^{\rm post}-\mu_i^{\rm true}$', \
							   fontsize=14)
		axes[k].legend(loc='upper right')
	if datafile is not None:
		axes[-1].set_xlim(wl[0], wl[-1])
		if window:
			axes[-1].set_xlabel(r'${\rm index}\,(i)$')
		else:
			axes[-1].set_xlabel(r'$\lambda-15100\,[{\rm Angstroms}]$')
			mp.setp(axes[-1].get_xticklabels(), rotation=45)
	else:
		axes[-1].set_xlabel(r'index ($i$)', fontsize=14)
	fig.subplots_adjust(hspace=0, wspace=0)
	mp.savefig('simple_test_mean.pdf', bbox_inches='tight')
	mp.close()

	# compare covariances
	if datafile is not None:
		fig, axes = mp.subplots(n_classes, 1, \
								figsize=(8, 5 * n_classes))	
		if n_classes == 1:
			axes = axis_to_axes(axes)
		for k in range(n_classes):
			min_cov, max_cov = np.min(mp_cov[:, :, k]), \
							   np.max(mp_cov[:, :, k])
			ext_cov = np.max((np.abs(min_cov), max_cov))
			cax = axes[k].matshow(mp_cov[:, :, k], vmin=-ext_cov, \
								  vmax=ext_cov, cmap=mpcm.seismic, \
								  interpolation = 'nearest')
			ax_pos = axes[k].get_position()
			cbar_ax = fig.add_axes([ax_pos.x0 + 0.04, ax_pos.y0, \
									0.02, ax_pos.y1 - ax_pos.y0])
			fig.colorbar(cax, cax=cbar_ax)
			#cbar = fig.colorbar(cax)
			axes[k].set_title(r'Mean Posterior')
			axes[k].tick_params(axis='both', which='both', \
								bottom='off', top='off', \
								labeltop='off', right='off', \
								left='off', labelleft='off')
	else:
		fig, axes = mp.subplots(n_classes, 3, \
								figsize=(16, 5 * n_classes))	
		if n_classes == 1:
			axes = axis_to_axes(axes)
		for k in range(n_classes):
			min_cov, max_cov = np.min(cov[:, :, k]), \
							   np.max(cov[:, :, k])
			ext_cov = np.max((np.abs(min_cov), max_cov))
			axes[k, 0].matshow(cov[:, :, k], vmin=-ext_cov, \
							   vmax=ext_cov, cmap=mpcm.seismic, \
							   interpolation='nearest')
			cax = axes[k, 1].matshow(mp_cov[:, :, k], vmin=-ext_cov, \
									 vmax=ext_cov, cmap=mpcm.seismic, \
									 interpolation = 'nearest')
			axes[k, 2].matshow(mp_cov[:, :, k] - cov[:, :, k], \
							   vmin=-ext_cov, vmax=ext_cov, \
							   cmap=mpcm.seismic, \
							   interpolation='nearest')
			fig.subplots_adjust(right=0.8)
			ax_pos = axes[k, 2].get_position()
			cbar_ax = fig.add_axes([0.84, ax_pos.y0, 0.02, \
									ax_pos.y1 - ax_pos.y0])
			fig.colorbar(cax, cax=cbar_ax)
			axes[k, 0].set_title(r'Ground Truth')
			axes[k, 1].set_title(r'Mean Posterior')
			axes[k, 2].set_title(r'Residual')
			for i in range(len(axes)):
				axes[k, i].tick_params(axis='both', which='both', \
									   bottom='off', top='off', \
									   labeltop='off', right='off', \
									   left='off', labelleft='off')
	mp.savefig('simple_test_covariance.pdf', bbox_inches='tight')
	mp.close()

	# compare correlations
	if datafile is not None:
		fig, axes = mp.subplots(n_classes, 1, \
								figsize=(8, 5 * n_classes))	
		if n_classes == 1:
			axes = axis_to_axes(axes)
		for k in range(n_classes):
			cax = axes[k].matshow(mp_cor[:, :, k], vmin=-1.0, \
								  vmax=1.0, cmap=mpcm.seismic, \
								  interpolation = 'nearest')
			ax_pos = axes[k].get_position()
			cbar_ax = fig.add_axes([ax_pos.x0 + 0.04, ax_pos.y0, \
									0.02, ax_pos.y1 - ax_pos.y0])
			fig.colorbar(cax, cax=cbar_ax)
			#cbar = fig.colorbar(cax)
			axes[k].set_title(r'Mean Posterior')
			axes[k].tick_params(axis='both', which='both', \
								bottom='off', top='off', \
								labeltop='off', right='off', \
								left='off', labelleft='off')
	else:
		fig, axes = mp.subplots(n_classes, 3, \
								figsize=(16, 5 * n_classes))	
		if n_classes == 1:
			axes = axis_to_axes(axes)
		for k in range(n_classes):
			axes[k, 0].matshow(cor[:, :, k], vmin=-1.0, \
							   vmax=1.0, cmap=mpcm.seismic, \
							   interpolation='nearest')
			cax = axes[k, 1].matshow(mp_cor[:, :, k], vmin=-1.0, \
									 vmax=1.0, cmap=mpcm.seismic, \
									 interpolation = 'nearest')
			axes[k, 2].matshow(mp_cor[:, :, k] - cor[:, :, k], \
							   vmin=-1.0, vmax=1.0, \
							   cmap=mpcm.seismic, \
							   interpolation='nearest')
			fig.subplots_adjust(right=0.8)
			ax_pos = axes[k, 2].get_position()
			cbar_ax = fig.add_axes([0.84, ax_pos.y0, 0.02, \
									ax_pos.y1 - ax_pos.y0])
			fig.colorbar(cax, cax=cbar_ax)
			axes[k, 0].set_title(r'Ground Truth')
			axes[k, 1].set_title(r'Mean Posterior')
			axes[k, 2].set_title(r'Residual')
			for i in range(len(axes)):
				axes[k, i].tick_params(axis='both', which='both', \
									   bottom='off', top='off', \
									   labeltop='off', right='off', \
									   left='off', labelleft='off')
	mp.savefig('simple_test_correlation.pdf', bbox_inches='tight')
	mp.close()

	# plot some realizations of the mean-posterior Gaussian process
	if n_gp_reals > 0:

		# one subfigure per class
		fig, axes = mp.subplots(n_classes, 1, \
								figsize=(8, 5 * n_classes), \
								sharex=True)	
		if n_classes == 1:
			axes = axis_to_axes(axes)
		for k in range(n_classes):

			# pick some samples at which to generate realizations
			i_sample = npr.randint(n_warmup, n_samples, n_gp_reals)
			gp_reals = np.zeros((n_gp_reals, n_bins))
			for i in range(n_gp_reals):
				gp_reals[i, :] = npr.multivariate_normal(mean_samples[:, k, i], \
														 cov_samples[:, :, k, i], \
														 1)
			ind_sort = np.argsort(gp_reals[:, 0])

			# plot, colouring in order of increasing first-bin value to aid
			# interpretation of correlations
			cols = [cm(x) for x in np.linspace(0.1, 0.9, n_gp_reals)]
			for i in range(n_gp_reals):
				axes[k].plot(wl, gp_reals[ind_sort[i], :], color=cols[i])
			axes[k].plot(wl, mp_mean[:, k], 'k')
			if datafile is not None and window:
				for i in range(n_windows):
					axes[k].axvline(wendices[i], color='k', lw=0.5, ls=':')
					axes[k].text(wendices[i], mp.gca().get_ylim()[0], \
								 wlabels[i], fontsize=8, ha='right', \
								 va='bottom')
			axes[k].set_ylabel(r'${\rm flux}$', fontsize=14)

		axes[-1].set_xlim(wl[0], wl[-1])
		if datafile is not None and window:
			axes[-1].set_xlabel(r'${\rm index}\,(i)$')
		else:
			axes[-1].set_xlabel(r'$\lambda-15100\,[{\rm Angstroms}]$')
			mp.setp(axes[-1].get_xticklabels(), rotation=45)
		mp.savefig('simple_test_gp_realizations.pdf', bbox_inches='tight')
		mp.close()

	# fun with ranks and eigenvalues of covariance matrices!
	n_eval_sig = np.zeros(n_classes, dtype=int)
	mp_cov_evals = np.zeros((n_bins, n_classes))
	mp_cov_evex = np.zeros((n_bins, n_bins, n_classes))
	fig, axes = mp.subplots(n_classes, 3, figsize=(16, 5 * n_classes))
	fig_e, axes_e = mp.subplots(1, 2, figsize=(16, 5))
	if precompress:
		fig_p, axes_p = mp.subplots(n_classes, 3, figsize=(16, 5 * n_classes))
	if n_classes == 1:
		axes = axis_to_axes(axes)
		axes_p = axis_to_axes(axes_p)
	cols = [cm(x) for x in np.linspace(0.1, 0.9, n_classes)]
	if precompress:
		axes_e[0].semilogy(d_evals, color='k', ls='--', \
						   label='data covariance')
		eval_bins = np.logspace(np.log10(d_evals[0]), \
								np.log10(d_evals[-1]), 20)
		pretty_hist(d_evals, eval_bins, axes_e[1], 'k', \
					fill=False, ls='--')
		cov_trunc_pca = \
			np.dot(d_evex[-n_pc_sig:, :].T, \
				   np.dot(np.diag(d_evals[-n_pc_sig:]), \
						  d_evex[-n_pc_sig:, :]))
	for k in range(n_classes):

		# calculate spectral decomposition and plot evals
		mp_cov_evals[:, k], mp_cov_evex[:, :, k] = \
			npl.eigh(mp_cov[:, :, k])
		ind_eval_sig = mp_cov_evals[:, k] > \
					   np.max(mp_cov_evals[:, k]) * eval_thresh
		n_eval_sig[k] = np.sum(ind_eval_sig)
		eval_bins = np.logspace(np.log10(mp_cov_evals[0, k]), \
								np.log10(mp_cov_evals[-1, k]), 20)
		axes_e[0].semilogy(mp_cov_evals[:, k], color=cols[k], \
						   label='class {:d}'.format(k + 1))
		pretty_hist(mp_cov_evals[:, k], eval_bins, axes_e[1], cols[k])
		axes_e[1].set_xscale('log')
		print 'MP covariance {:d} '.format(k) + \
			  'rank: {:d}'.format(npl.matrix_rank(mp_cov[:, :, k]))
		print '{:d} significant evals'.format(n_eval_sig[k])

		# restrict rank to most significant evex
		mp_cov_evals[~ind_eval_sig, k] = 0.0
		mp_cov_low_rank = np.dot(mp_cov_evex[:, :, k], \
								 np.dot(np.diag(mp_cov_evals[:, k]), \
								 		mp_cov_evex[:, :, k].T))
		ext_cov = np.max((np.abs(mp_cov[:, :, k]), mp_cov[:, :, k]))

		# plot comparison between full and low-rank covariances
		axes[k, 0].matshow(mp_cov[:, :, k], vmin=-ext_cov, \
						   vmax=ext_cov, cmap=mpcm.seismic, \
						   interpolation='nearest')
		cax = axes[k, 1].matshow(mp_cov_low_rank, vmin=-ext_cov, \
								 vmax=ext_cov, cmap=mpcm.seismic, \
								 interpolation = 'nearest')
		axes[k, 2].matshow(mp_cov_low_rank - mp_cov[:, :, k], \
						   vmin=-ext_cov, vmax=ext_cov, \
						   cmap=mpcm.seismic, interpolation='nearest')
		fig.subplots_adjust(right=0.8)
		ax_pos = axes[k, 2].get_position()
		cbar_ax = fig.add_axes([0.84, ax_pos.y0, 0.02, \
								ax_pos.y1 - ax_pos.y0])
		fig.colorbar(cax, cax=cbar_ax)
		axes[k, 0].set_title(r'Mean Posterior')
		axes[k, 1].set_title('Mean Posterior (rank ' + \
							 '{:d})'.format(n_eval_sig[k]))
		axes[k, 2].set_title(r'Residual')
		for i in range(len(axes)):
			axes[k, i].tick_params(axis='both', which='both', \
								   bottom='off', top='off', \
								   labeltop='off', right='off', \
								   left='off', labelleft='off')

		# plot comparison between PCA and MAP covariances
		if precompress:
			axes_p[k, 0].matshow(mp_cov[:, :, k], vmin=-ext_cov, \
							   vmax=ext_cov, cmap=mpcm.seismic, \
							   interpolation='nearest')
			cax = axes_p[k, 1].matshow(cov_trunc_pca, vmin=-ext_cov, \
									 vmax=ext_cov, cmap=mpcm.seismic, \
									 interpolation='nearest')
			axes_p[k, 2].matshow(mp_cov_low_rank - cov_trunc_pca, \
							   vmin=-ext_cov, vmax=ext_cov, \
							   cmap=mpcm.seismic, interpolation='nearest')
			axes_p[k, 1].set_title(r'Truncated PCA')
			fig_p.subplots_adjust(right=0.8)
			ax_pos = axes_p[k, 2].get_position()
			cbar_ax = fig_p.add_axes([0.84, ax_pos.y0, 0.02, \
									ax_pos.y1 - ax_pos.y0])
			fig_p.colorbar(cax, cax=cbar_ax)
			axes_p[k, 0].set_title(r'Mean Posterior')
			axes_p[k, 1].set_title('Truncated PCA (rank ' + \
								   '{:d})'.format(n_pc_sig))
			axes_p[k, 2].set_title(r'Residual')
			for i in range(len(axes_p)):
				axes_p[k, i].tick_params(axis='both', which='both', \
									   bottom='off', top='off', \
									   labeltop='off', right='off', \
									   left='off', labelleft='off')

	# finish plots
	axes_e[0].legend(loc='upper left')
	axes_e[0].set_xlabel(r'${\rm index }\,i$')
	axes_e[0].set_ylabel(r'$\lambda_i$')
	axes_e[0].set_xlim(0, n_bins)
	axes_e[1].set_xlabel(r'$\lambda_i$')
	axes_e[1].set_ylabel(r'$N(\lambda_i)$')
	fig.savefig('simple_test_low_rank_covariance.pdf', bbox_inches='tight')
	fig_e.savefig('simple_test_evals.pdf', bbox_inches='tight')
	mp.close(fig)
	mp.close(fig_e)
	if precompress:
		fig_p.savefig('simple_test_pca_vs_map.pdf', bbox_inches='tight')
		mp.close(fig_p)

	# condition numbers of sampled covariance matrices
	if sample:
		for k in range(n_classes):
			mp.semilogy(conds[k, :])
			if datafile is None:
				mp.axhline(npl.cond(cov[:, :, k]))
		mp.xlabel('index')
		mp.ylabel('condition number')
		mp.savefig('simple_test_conds.pdf', bbox_inches='tight')
		mp.close()

	# eigenvectors of interest!
	n_plot_max = min(np.max(n_eval_sig), 10)
	fig, axes = mp.subplots(n_plot_max, n_classes, \
							figsize=(8 * n_classes, 3 * n_plot_max), \
							sharex=True)
	if n_classes == 1:
		axes = axis_to_axes(axes, True)
	for k in range(n_classes):
		n_plot = min(n_eval_sig[k], 10)
		for i in range(n_plot):
			axes[i, k].plot(wl, mp_cov_evex[:, -1 - i, k])
			if datafile is not None and window:
				for j in range(n_windows):
					axes[i, k].axvline(wendices[j], color='k', lw=0.5, \
									   ls=':')
					axes[i, k].text(wendices[j], axes[i, k].get_ylim()[0], \
									wlabels[j], fontsize=8, ha='right', \
									va='bottom')
			label = r'$\lambda_' + '{:d} = '.format(i) + r'{\rm ' + \
					'{:9.2e}'.format(mp_cov_evals[-1 - i, k]) + r'}$'
			axes[i, k].text(0.99, 0.95, label, transform=axes[i, k].transAxes, \
						 fontsize=16, ha='right', va='top')
			axes[i, k].set_ylabel(r'$v_{i' + '{:d}'.format(i) + r'}$', \
								  fontsize=16)
		axes[-1, k].set_xlabel(r'${\rm index},\,i$', fontsize=16)
		axes[-1, k].set_xlim(wl[0], wl[-1])
	fig.subplots_adjust(hspace=0)
	mp.savefig('simple_test_evex.pdf', bbox_inches='tight')
	mp.close()

	# conditional variance plots: which features best predict others?
	# this might produce the most gigantic plots ever...
	if datafile is not None and window:

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

				# find indices
				if i == 0:
					inds_i = (wl < wendices[i])
					inds_o = (wl >= wendices[i])
				else:
					inds_i = (wl >= wendices[i - 1]) & (wl < wendices[i])
					inds_o = (wl < wendices[i - 1]) | (wl >= wendices[i])
				n_i = np.sum(inds_i)
				n_o = np.sum(inds_o)

				# construct submatrices
				s_ii = np.zeros((n_i, n_i))
				s_io = np.zeros((n_i, n_o))
				s_oo = np.zeros((n_o, n_o))
				n = 0
				for j in range(n_bins):
					if inds_i[j]:
						s_ii[n, :] = mp_cov[j, inds_i, k]
						s_io[n, :] = mp_cov[j, inds_o, k]
						n += 1
					else:
						s_oo[j - n, :] = mp_cov[j, inds_o, k]

				# calculate and plot covariance of conditional distribution
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
		fig.savefig('simple_test_conditional_stddevs.pdf', bbox_inches='tight')
		mp.close(fig)
		fig_d.subplots_adjust(hspace=0)
		fig_d.savefig('simple_test_conditional_inf_gain.pdf', bbox_inches='tight')
		mp.close(fig_d)
		fig_e.subplots_adjust(hspace=0)
		fig_e.savefig('simple_test_most_correlated_stddevs.pdf', bbox_inches='tight')
		mp.close(fig_e)

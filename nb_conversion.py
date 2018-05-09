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
import corner as co

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


# @TODO LIST
#  - multiple classes
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
n_bins = 7 # 50
n_spectra = 2000
n_classes = 2
n_samples = 500 # 1000
n_warmup = n_samples / 4
n_gp_reals = 50
jeffreys_prior = 1
diagnose = False
datafile = None # 'data/redclump_1_alpha_nonorm.h5' # filename or None
window = True
inf_noise = 1.0e5
reg_noise = 1.0e-6

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
	f = h5py.File('data/redclump_1_alpha_nonorm.h5','r')
	full_data = f['dataset_1'][:]
	f.close()

	# construct data vector and noise covariance: mask handled by 
	# noise variance
	if window:

		# read in window definitions. file contains elements with 
		# positions of features within three wavelength ranges. take 
		# windows of +/- 2.5 Angstroms about each line center; centers
		# of 999 Angstroms should be ignored
		wl = full_data[:, 0, 0]
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
		#wl = wl[windices]
		wl = np.arange(n_bins)
		data = full_data[windices, 0: n_spectra, 1].T
		var_noise = full_data[windices, 0: n_spectra, 2].T ** 2
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

	else:

		i_min = 1770 # 1785
		if rank == 0:
			msg = 'selecting wavelengths in range {0:.2f}-{1:.2f} Angstroms'
			print msg.format(full_data[i_min, 0, 0], \
							 full_data[i_min + n_bins, 0, 0])
		wl = full_data[i_min: i_min + n_bins, 0, 0].T - 15100.0
		data = full_data[i_min: i_min + n_bins, 0: n_spectra, 1].T
		var_noise = full_data[i_min: i_min + n_bins, 0: n_spectra, 2].T ** 2
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
	in_class_k = (class_ids == k)
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
		axes[-1].set_xticks(rotation=45)
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
			axes[-1].set_xlabel(r'$\lambda-15100\,[{\rm Angstroms}]$')
			axes[-1].set_xticks(rotation=45)
		else:
			axes[-1].set_xlabel(r'${\rm index}\,(i)$')
		mp.savefig('simple_test_gp_realizations.pdf', bbox_inches='tight')
		mp.close()

	# fun with ranks and eigenvalues of covariance matrices!
	n_eval_sig = np.zeros(n_classes, dtype=int)
	mp_cov_evals = np.zeros((n_bins, n_classes))
	mp_cov_evex = np.zeros((n_bins, n_bins, n_classes))
	fig, axes = mp.subplots(n_classes, 3, figsize=(16, 5 * n_classes))	
	if n_classes == 1:
		axes = axis_to_axes(axes)
	for k in range(n_classes):
		mp_cov_evals[:, k], mp_cov_evex[:, :, k] = npl.eigh(mp_cov[:, :, k])
		ind_eval_sig = mp_cov_evals[:, k] > \
					   np.max(mp_cov_evals[:, k]) / 1.0e2
		n_eval_sig[k] = np.sum(ind_eval_sig)
		print 'MP covariance {:d} '.format(k) + \
			  'rank: {:d}'.format(npl.matrix_rank(mp_cov[:, :, k]))
		print '{:d} significant evals'.format(n_eval_sig[k])
		mp_cov_evals[~ind_eval_sig, k] = 0.0
		mp_cov_low_rank = np.dot(mp_cov_evex[:, :, k], \
								 np.dot(np.diag(mp_cov_evals[:, k]), \
								 		mp_cov_evex[:, :, k].T))
		ext_cov = np.max((np.abs(mp_cov[:, :, k]), mp_cov[:, :, k]))
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
	mp.savefig('simple_test_low_rank_covariance.pdf', bbox_inches='tight')
	mp.close()

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


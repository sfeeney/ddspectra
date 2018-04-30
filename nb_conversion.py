import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import scipy.stats as sps
import h5py
import os
if 'DISPLAY' not in os.environ.keys():
	import matplotlib
	matplotlib.use('Agg')
import matplotlib.pyplot as mp
import matplotlib.cm as mpcm
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
		target = np.zeros(target_distrib.shape)
		mpi.COMM_WORLD.Reduce(target_distrib, target, op=mpi.SUM, \
							  root=0)
		#mpi.COMM_WORLD.Allreduce(target_distrib, target, op=mpi.SUM)
	else:
		target = target_distrib
	return target


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
n_bins = 100 # 7 # 50
n_spectra = 2000
n_samples = 1000
n_warmup = n_samples / 4
n_gp_reals = 50
jeffreys_prior = 1
diagnose = False
datafile = 'data/redclump_1_alpha_nonorm.h5' # filename or None
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
	mean = np.zeros(n_bins)
	cov = np.zeros((n_bins, n_bins))
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

			# generate a covariance matrix
			norm_vars = npr.randn(n_bins)
			in_mat = np.zeros((n_bins, n_bins))
			for i in range(n_bins):
				in_mat[:, i] = norm_vars
			u, s, v = npl.svd(in_mat)
			evex = u
			evals = np.linspace(1.0, n_bins, n_bins) ** -5
			#evals = np.linspace(1.0, n_bins, n_bins) ** -2
			#evals = np.linspace(1.0, n_bins, n_bins)
			cov = np.dot(np.dot(evex, np.diag(evals)), evex.T)
			#cov = np.diag(np.diag(cov))
			evals, evex = npl.eigh(cov)

			# generate a mean vector
			mean = 1.0 + 4.0 * (npr.rand(n_bins) - 0.5)
		print 'input covariance condition number ', npl.cond(cov)

		# optionally plot test inputs
		if diagnose:
			mp.imshow(cov, interpolation='nearest')
			mp.show()
			mp.plot(mean)
			mp.show()

		# generate signals from this model
		spectra_true = npr.multivariate_normal(mean, cov, n_spectra)
		if diagnose:
			print spectra_true.shape
			for i in range(n_spectra):
				mp.plot(spectra_true[i, :], 'k-', alpha=0.3)
			mp.show()

		# generate noise and mask
		var_noise = npr.uniform(0.0, 0.2, n_bins)
		cov_noise = np.diag(var_noise)
		inv_cov_noise = npl.inv(cov_noise)      # inverseNoiseCov
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
		centers = []
		for i in range(len(wdata)):
			#for j in range(3):
			for j in range(2):
				center = wdata[i][j + 1]
				if center != 999.0:
					centers.append(center)
		centers = np.sort(centers)
		windices = np.full(len(wl), False, dtype=bool)
		if rank == 0:
			msg = 'selecting wavelengths within 2.5 Angstroms of:'
		for i in range(len(centers)):
			windices = np.logical_or(windices, (wl >= centers[i] - 2.5) & \
											   (wl <= centers[i] + 2.5))	
			if rank == 0:
				msg = '{0:d}: {1:.2f} Angstroms'
				print msg.format(i, centers[i])
		'''mp.plot(wl, windices)
		for i in range(len(centers)):
			mp.axvline(centers[i],color='k')
		mp.show()
		exit()'''

		# select data
		n_bins = np.sum(windices)
		wl = wl[windices]
		data = full_data[windices, 0: n_spectra, 1].T
		var_noise = full_data[windices, 0: n_spectra, 2].T ** 2
		inv_cov_noise = np.zeros((n_spectra, n_bins, n_bins))
		for i in range(n_spectra):
			inv_cov_noise[i, :, :] = np.diag(1.0 / var_noise[i, :])

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
		mp.subplots_adjust(bottom=0.15)
		mp.savefig('simple_test_apogee_inputs.pdf', \
				   bbox_inches='tight')
		mp.show()

# initial conditions for sampler
mean_sample = np.mean(data, 0)          # signalpriormean
cov_sample = np.cov(data, rowvar=False) # SignalCovariance
mean_samples = np.zeros((n_bins, n_samples))
cov_samples = np.zeros((n_bins, n_bins, n_samples))
conds = np.zeros(n_samples)

# Gibbs sample!
job_list = allocate_jobs(n_spectra, n_procs, rank)
d_sample = n_samples / 10
for i in range(n_samples):

	# report progress
	if np.mod(i, d_sample) == 0 and rank == 0:
		print i, '/', n_samples

	# invert current sample covariance
	if not no_s_inv:
		inv_cov_sample = npl.inv(cov_sample)

	# calculate WF for each spectrum and use to draw true spectra
	spectra_samples = np.zeros(data.shape)
	for j in job_list:

		# avoid S^-1?
		if no_s_inv:

			# check if have spectrum-dependent noise
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
			if len(inv_cov_noise.shape) == 3:
				inv_cov_noise_j = inv_cov_noise[j, :, :]
			else:
				inv_cov_noise_j = np.outer(mask[j, :], mask[j, :]) * \
								  inv_cov_noise
			cov_wf = npl.inv(inv_cov_sample + inv_cov_noise_j)
			cov_wf = symmetrize(cov_wf)
			mean_wf = np.dot(cov_wf, \
							 np.dot(inv_cov_sample, mean_sample) + \
							 np.dot(inv_cov_noise_j, data[j, :]))
			
		spectra_samples[j, :] = npr.multivariate_normal(mean_wf, \
														cov_wf, 1)
	spectra_samples = complete_array(spectra_samples, use_mpi)

	# sample everything else on master process to avoid race 
	# conditions
	if rank == 0:

		# sample signal mean
		mean_sample = \
			npr.multivariate_normal(np.mean(spectra_samples, 0), \
									cov_sample / n_spectra, 1)[0, :]

		# sample signal covariance matrix
		# NB: scipy.stats uses numpy.random seed, which i've already set
		n_dof = n_spectra - (1 - jeffreys_prior) * (n_bins + 1)
		sigma = np.zeros((n_bins, n_bins))
		for j in range(n_spectra):
			delta = spectra_samples[j, :] - mean_sample
			sigma += np.outer(delta, delta)
		cov_sample = sps.invwishart.rvs(n_dof, sigma, 1) + \
					 np.diag(np.ones(n_bins) * reg_noise)

		# store samples (marginalize over true spectra)
		mean_samples[:, i] = mean_sample
		cov_samples[:, :, i] = cov_sample
        
	conds[i] = npl.cond(cov_sample)
	if rank == 0:
		print conds[i]

	# broadcast required objects to all processes
	if use_mpi:
		mpi.COMM_WORLD.Bcast(mean_sample, root=0)
		mpi.COMM_WORLD.Bcast(cov_sample, root=0)

# summarize results
mp_mean = np.mean(mean_samples[:, n_warmup:], 1)
sdp_mean = np.std(mean_samples[:, n_warmup:], 1)
mp_cov = np.mean(cov_samples[:, :, n_warmup:], 2)
if rank == 0:

	# selection of trace plots
	fig, axes = mp.subplots(3, 1, figsize=(8, 5), sharex=True)
	axes[0].plot(mean_samples[0, :])
	axes[1].plot(mean_samples[n_bins / 2, :])
	axes[2].plot(mean_samples[-1, :])
	axes[2].set_xlabel('sample')
	axes[1].set_ylabel('mean')
	fig.subplots_adjust(hspace=0, wspace=0)
	mp.savefig('simple_test_trace.pdf', bbox_inches='tight')
	mp.close()

	# compare true and sampled means
	if diagnose:
		for i in range(n_samples):
			mp.plot(mean_samples[:, i], 'k-', alpha=0.3)
		if datafile is None:
			mp.plot(mean, 'r-', alpha=1.0)
		mp.plot(mp_mean, 'y--', alpha=1.0)
		mp.show()

	# as above but with reference to noise and posterior standard deviations
	if datafile is not None:
		mp.fill_between(wl, mp_mean - sdp_mean, \
						mp_mean + sdp_mean, color='LightGrey', \
						label=r'posterior $\sigma$')
		mp.plot(wl, mp_mean, 'r', label='posterior mean')
	else:
		#mp.fill_between(range(n_bins), -np.sqrt(var_noise), \
		#				np.sqrt(var_noise), color='grey', \
		#				label=r'noise $\sigma$')
		mp.fill_between(range(n_bins), mp_mean - mean - sdp_mean, \
						mp_mean - mean + sdp_mean, color='LightGrey', \
						label=r'posterior $\sigma$')
		mp.plot([0, n_bins - 1], [0.0, 0.0], 'k--')
		mp.plot(mp_mean - mean, 'r', label='residual')
	if datafile is not None:
		mp.xlim(wl[0], wl[-1])
		mp.xlabel(r'$\lambda-15100\,[{\rm Angstroms}]$')
		mp.ylabel(r'$\mu^{\rm post}$', fontsize=14)
		mp.xticks(rotation=45)
	else:
		mp.xlabel(r'index ($i$)', fontsize=14)
		mp.ylabel(r'$\mu_i^{\rm post}-\mu_i^{\rm true}$', fontsize=14)
	mp.legend(loc='upper right')
	mp.savefig('simple_test_mean.pdf', bbox_inches='tight')
	mp.close()

	# compare covariances
	if datafile is not None:
		min_cov, max_cov = np.min(mp_cov), np.max(mp_cov)
		ext_cov = np.max((np.abs(min_cov), max_cov))
		fig, ax = mp.subplots(1, 1, figsize=(8, 5))
		cax = ax.matshow(mp_cov, vmin=-ext_cov, vmax=ext_cov, \
						 cmap=mpcm.seismic, interpolation = 'nearest')
		cbar = fig.colorbar(cax)
		ax.set_title(r'Mean Posterior')
		ax.tick_params(axis='both', which='both', bottom='off', \
					   top='off', labeltop='off', right='off', \
					   left='off', labelleft='off')
	else:
		min_cov, max_cov = np.min(cov), np.max(cov)
		ext_cov = np.max((np.abs(min_cov), max_cov))
		fig, axes = mp.subplots(1, 3, figsize=(16, 5))
		axes[0].matshow(cov, vmin=-ext_cov, vmax=ext_cov, \
						cmap=mpcm.seismic, interpolation='nearest')
		cax = axes[1].matshow(mp_cov, vmin=-ext_cov, vmax=ext_cov, \
							  cmap=mpcm.seismic, interpolation = 'nearest')
		axes[2].matshow(mp_cov - cov, vmin=-ext_cov, vmax=ext_cov, \
						cmap=mpcm.seismic, interpolation='nearest')
		fig.subplots_adjust(right=0.8)
		cbar_ax = fig.add_axes([0.84, 0.18, 0.02, 0.64])
		fig.colorbar(cax, cax=cbar_ax)
		axes[0].set_title(r'Ground Truth')
		axes[1].set_title(r'Mean Posterior')
		axes[2].set_title(r'Residual')
		for i in range(len(axes)):
			axes[i].tick_params(axis='both', which='both', bottom='off', \
								top='off', labeltop='off', right='off', \
								left='off', labelleft='off')
	mp.savefig('simple_test_covariance.pdf', bbox_inches='tight')
	mp.close()

	# plot some realizations of the mean-posterior Gaussian process
	if n_gp_reals > 0:

		# pick some samples at which to generate realizations
		#gp_reals = npr.multivariate_normal(mp_mean, mp_cov, n_gp_reals)
		i_sample = npr.randint(n_warmup, n_samples, n_gp_reals)
		gp_reals = np.zeros((n_gp_reals, n_bins))
		for i in range(n_gp_reals):
			gp_reals[i, :] = npr.multivariate_normal(mean_samples[:, i], \
													 cov_samples[:, :, i], \
													 1)
		ind_sort = np.argsort(gp_reals[:, -1])

		# plot, colouring in order of increasing first-bin value to aid
		# interpretation of correlations
		cols = [cm(x) for x in np.linspace(0.1, 0.9, n_gp_reals)]
		mp.fill_between(range(n_bins), mp_mean - sdp_mean, \
						mp_mean + sdp_mean, color='LightGrey')
		for i in range(n_gp_reals):
			mp.plot(wl, gp_reals[ind_sort[i], :], color=cols[i])
		mp.plot(wl, mp_mean, 'k')
		mp.xlim(wl[0], wl[-1])
		mp.xlabel(r'$\lambda-15100\,[{\rm Angstroms}]$')
		mp.ylabel(r'${\rm flux}$', fontsize=14)
		mp.xticks(rotation=45)
		mp.savefig('simple_test_gp_realizations.pdf', bbox_inches='tight')
		mp.close()

	# takes forever if too many bins...
	if n_bins < 10:

		# check correlations between binned mean estimates
		labels = [r'$\mu_{' + '{:d}'.format(i) + '}$' for i in range(n_bins)]
		fig = co.corner(mean_samples.T, bins=10, labels=labels, \
						plot_density=False, plot_contours=False, \
						no_fill_contours=True)
		for i in range(n_bins):
			for j in range(i):

				a_ind = i * n_bins + j
				fig.axes[a_ind].axhline(mean[i], \
										color='red', ls=':', \
										lw = 1.5)
			for j in range(i + 1):
				a_ind = i * n_bins + j
				fig.axes[a_ind].axvline(mean[j], \
										color='red', ls=':', \
										lw = 1.5)
		mp.savefig('simple_test_mean_posterior.pdf')
		mp.close()

	# fun with ranks of covariance matrices!
	mp_cov_evals, mp_cov_evex = npl.eigh(mp_cov)
	ind_eval_sig = mp_cov_evals > np.max(mp_cov_evals) / 1.0e2
	n_eval_sig = np.sum(ind_eval_sig)
	print 'MP covariance rank: {:d}'.format(npl.matrix_rank(mp_cov))
	print 'MP covariance evals:'
	print mp_cov_evals
	print '{:d} significant evals'.format(n_eval_sig)
	mp_cov_evals[~ind_eval_sig] = 0.0
	mp_cov_low_rank = np.dot(mp_cov_evex, \
							 np.dot(np.diag(mp_cov_evals), \
							 		mp_cov_evex.T))
	ext_cov = np.max((np.abs(mp_cov), mp_cov))
	fig, axes = mp.subplots(1, 3, figsize=(16, 5))
	axes[0].matshow(mp_cov, vmin=-ext_cov, vmax=ext_cov, \
					cmap=mpcm.seismic, interpolation='nearest')
	cax = axes[1].matshow(mp_cov_low_rank, vmin=-ext_cov, vmax=ext_cov, \
						  cmap=mpcm.seismic, interpolation = 'nearest')
	axes[2].matshow(mp_cov_low_rank - mp_cov, vmin=-ext_cov, vmax=ext_cov, \
					cmap=mpcm.seismic, interpolation='nearest')
	fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.84, 0.18, 0.02, 0.64])
	fig.colorbar(cax, cax=cbar_ax)
	axes[0].set_title(r'Mean Posterior')
	axes[1].set_title('Mean Posterior (rank {:d})'.format(n_eval_sig))
	axes[2].set_title(r'Residual')
	for i in range(len(axes)):
		axes[i].tick_params(axis='both', which='both', bottom='off', \
							top='off', labeltop='off', right='off', \
							left='off', labelleft='off')
	mp.savefig('simple_test_low_rank_covariance.pdf', bbox_inches='tight')
	mp.close()

	# condition numbers of sampled covariance matrices
	mp.plot(conds)
        if datafile is None:
                mp.axhline(npl.cond(cov))
	mp.xlabel('index')
	mp.ylabel('condition number')
	mp.savefig('simple_test_conds.pdf', bbox_inches='tight')
	mp.close()

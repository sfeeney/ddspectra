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

def allocate_all_jobs(n_jobs, n_procs=1):
	allocation = []
	n_j_allocated = 0
	for i in range(n_procs):
		n_j_remain = n_jobs - n_j_allocated
		n_p_remain = n_procs - i
		n_j_to_allocate = n_j_remain / n_p_remain
		allocation.append(range(n_j_allocated, \
								n_j_allocated + n_j_to_allocate))
		n_j_allocated += n_j_to_allocate
	return allocation

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

def complete_array_alt(target_distrib, job_lists, use_mpi=False, \
					   last=True):
	if use_mpi:

		# determine required dimensions
		target_shape = target_distrib.shape
		n_dim = len(target_shape)
		if last:
			i_bcast = n_dim - 1
		else:
			i_bcast = 0
		dim = [i for i in range(n_dim) if i != i_bcast]
		shape = [target_shape[i] for i in dim]

		# dimension output and work arrays
		target = np.zeros(target_shape, dtype=target_distrib.dtype)
		bcast = np.zeros(shape, dtype=target_distrib.dtype)

		# loop and broadcast by array rows/columns
		n_procs = len(job_lists)
		for i in range(n_procs):
			for j in job_lists[i]:
				if rank == i:
					if last:
						bcast[...] = target_distrib[..., j]
					else:
						bcast[...] = target_distrib[j, ...]
				mpi.COMM_WORLD.Bcast(bcast, root=i)
				if last:
					target[..., j] = bcast[...]
				else:
					target[j, ...] = bcast[...]

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
				ls='-', zorder=None, label=None):

	hist, bin_edges = np.histogram(data, bins=bins, density=density)
	bins_to_plot = np.append(bins, bins[-1])
	hist_to_plot = np.append(np.insert(hist, 0, 0.0), 0.0)
	if zorder is not None:
		if label is not None:
			axis.step(bins_to_plot, hist_to_plot, where='pre', \
					  color=color, linestyle=ls, zorder=zorder, \
					  label=label)
		else:
			axis.step(bins_to_plot, hist_to_plot, where='pre', \
					  color=color, linestyle=ls, zorder=zorder)
		if fill:
			axis.fill_between(bins_to_plot, hist_to_plot, \
							  color=color, alpha=0.7, step='pre', \
							  zorder=zorder)
	else:
		if label is not None:
			axis.step(bins_to_plot, hist_to_plot, where='pre', \
					  color=color, linestyle=ls, label=label)
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
recovery_test = False
constrain = False
no_s_inv = False
sample = True
precompress = False
inpaint = False
n_bins = 7 # 50
n_spectra = 29502
n_classes = 2
n_samples = 500 # 1000
n_warmup = n_samples / 4
n_gp_reals = 50
jeffreys_prior = 0
diagnose = False
datafile = 'data/redclump_{:d}_alpha_nonorm.h5' # filename or None
window = 'data/centers_subset2_ce_nd.txt' # filename or None
save_spectra = 'data/ids_ce_nd_1_fully_masked_lowest_10_snr.txt' # filename or None
inf_noise = 1.0e5
reg_noise = 1.0e-6
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
if jeffreys_prior == 0:
	io_base += 'no_jp_'
if recovery_test:
	io_base += 'rec_test_'
	i_rec_test = 0
	j_rec_test_lo = 10
	j_rec_test_hi = 30

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
	seed = npr.randint(221216, 231014, 1)[0]
npr.seed(seed)

# assign lists of spectral and class IDs for each MPI process
job_lists = allocate_all_jobs(n_spectra, n_procs)
class_job_lists = allocate_all_jobs(n_classes, n_procs)

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
		class_probs = npr.dirichlet(np.ones(n_classes) * 10.0, 1)[0]
		class_ids = npr.choice(n_classes, n_spectra, p=class_probs)
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

	# file io on master only
	if rank == 0:

		# data may be distributed over multiple input files. if we're 
		# windowing we don't know a priori how many spectral bins are
		# going to be selected. read in the first datafile and 
		# perform this windowing to ensure we can broadcast to other 
		# processes
		n_to_load = n_spectra
		n_file = 1
		wl, full_data = read_spectra(n_to_load, \
									 datafile.format(n_file), \
									 return_wl=True)

		# determine data selection
		if window is not None:

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

		else:

			# some portions of the spectra are masked in all objects.
			# remove them.
			all_masked = np.array([[0, 247], [3273, 3688], \
								  [6079, 6371], [8334, 8575]])
			n_all_masked = all_masked.shape[0]
			print 'removing totally masked wavelengths in ranges:'
			windices = np.full(len(wl), True, dtype=bool)
			msg = '{0:1d}: {1:.2f}-{2:.2f} A'
			for i in range(n_all_masked):
				print msg.format(i, wl[all_masked[i, 0]], \
								 wl[all_masked[i, 1] - 1])
				windices[all_masked[i, 0]: all_masked[i, 1]] = False
			wlabels = ['{:1d}'.format(i) for i in \
					   range(1, n_all_masked)]

		# determine the selected-data indices where the breaks are
		n_bins = np.sum(windices)
		dwindices = np.append(0, windices[1:].astype(int) - \
								 windices[:-1].astype(int))
		wendows = [x[0] for x in np.argwhere(dwindices < 0)]
		n_windows = len(wendows)
		wendices = []
		for i in range(n_windows):
			wendices.append(np.sum(windices[0: wendows[i]]))
		n_in_bin = np.append(wendices[0], np.diff(wendices))
		wl = np.arange(n_bins)

		# select data
		data = np.zeros((n_spectra, n_bins))
		var_noise = np.zeros((n_spectra, n_bins))
		n_loaded = 0
		while True:

			# load in existing data
			n_in_file = full_data.shape[1]
			data[n_loaded: n_loaded + n_in_file, :] = \
				full_data[windices, :, 0].T
			var_noise[n_loaded: n_loaded + n_in_file, :] = \
				full_data[windices, :, 1].T ** 2
			#for i in range(n_loaded, n_loaded + n_in_file):
			#	inv_cov_noise[i, :, :] = np.diag(1.0 / var_noise[i, :])
			n_to_load -= n_in_file
			n_loaded += n_in_file

			# search for more data if needed
			if n_to_load > 0:
				n_file += 1
				full_data = read_spectra(n_to_load, \
										 datafile.format(n_file))
			else:
				break

		# optionally mask a section of one star's spectrum 
		# artificially to test method. store complete input spectrum
		# for comparison
		if recovery_test:

			rec_test_data = np.zeros(n_bins)
			rec_test_var_noise = np.zeros(n_bins)
			rec_test_data[:] = data[i_rec_test, :]
			rec_test_var_noise[:] = var_noise[i_rec_test, :]
			data[i_rec_test, j_rec_test_lo: j_rec_test_hi] = 1.0
			var_noise[i_rec_test, j_rec_test_lo: j_rec_test_hi] = 4.0e6

		# optionally inpaint masked regions with masked mean to 
		# improve PCA
		full_data = np.zeros((n_spectra, n_bins))
		if inpaint:
			masked = np.sqrt(var_noise) > 1000.0
			masked_mean = (np.sum(data, 0) - np.sum(masked, 0)) / \
						  np.sum(~masked, 0)
			for i in range(n_spectra):
				full_data[i, ~masked[i, :]] = data[i, ~masked[i, :]]
				full_data[i, masked[i, :]] = masked_mean[masked[i, :]]
		else:
			full_data[:, :] = data[:, :]

		# plot if desired
		if diagnose:
			n_to_plot = 20 # n_spectra
			cols = [cm(x) for x in np.linspace(0.1, 0.9, n_to_plot)]
			fig, axes = mp.subplots(1, 2, figsize=(16, 5))
			for i in range(n_to_plot):
				axes[0].plot(wl, data[i, :], color=cols[i], alpha=0.5)
				axes[1].semilogy(wl, np.sqrt(var_noise[i, :]), \
								 color=cols[i], alpha=0.5)
			for ax in axes:
				ax.set_xlim(wl[0], wl[-1])
				if window is not None:
					ax.set_xlabel(r'${\rm index},\,i$')
				else:
					ax.set_xlabel(r'$\lambda-15100\,[{\rm Angstroms}]$')
					mp.setp(ax.get_xticklabels(), rotation=45)
			axes[0].set_ylabel(r'${\rm flux}$')
			axes[1].set_ylabel(r'$\sigma_{\rm flux}$')
			if window is not None:
				for i in range(n_windows):
					axes[0].axvline(wendices[i], color='k', lw=0.5)
					axes[1].axvline(wendices[i], color='k', lw=0.5)
					x_text = n_in_bin[i] / 2.0
					if i > 0:
						x_text += wendices[i - 1]
					axes[0].text(x_text, axes[0].get_ylim()[0], \
								 wlabels[i], fontsize=8, \
								 ha='center', va='bottom')
					axes[1].text(x_text, axes[1].get_ylim()[0], \
								 wlabels[i], fontsize=8, \
								 ha='center', va='bottom')
			mp.subplots_adjust(bottom=0.15)
			mp.savefig(io_base + 'apogee_inputs.pdf', \
					   bbox_inches='tight')
			mp.show()

		# read in IDs of spectra for which to save samples
		if save_spectra is not None:
			save_spectra_ids = np.genfromtxt(save_spectra).astype(int)
			n_save_spectra = len(save_spectra_ids)
	
	'''
	# ensure all processes have same data
	if use_mpi:

		# initialize data arrays on slave processes
		n_bins = mpi.COMM_WORLD.bcast(n_bins, root=0)
		if rank > 0:
			if window is not None:
				wl = np.zeros(n_bins, dtype=int)
			else:
				wl = np.zeros(n_bins)
			data = np.zeros((n_spectra, n_bins))
			var_noise = np.zeros((n_spectra, n_bins))
		inv_cov_noise = None
		mpi.COMM_WORLD.Bcast(wl, root=0)
		mpi.COMM_WORLD.Bcast(data, root=0)
		mpi.COMM_WORLD.Bcast(var_noise, root=0)
	'''
			
	# ensure all processes have appropriate data
	if use_mpi:

		# initialize data arrays on slave processes
		n_bins = mpi.COMM_WORLD.bcast(n_bins, root=0)
		if rank > 0:
			if window is not None:
				wl = np.zeros(n_bins, dtype=int)
			else:
				wl = np.zeros(n_bins)
			data = np.zeros((len(job_lists[rank]), n_bins))
			var_noise = np.zeros((len(job_lists[rank]), n_bins))
			full_data = np.zeros((n_spectra, n_bins))
		inv_cov_noise = None

		# share data and noise variances such that only one 
		# distributed copy is present. need to keep a full copy of 
		# the data for now in order to calculate its PCA and 
		# initialize the sampler
		mpi.COMM_WORLD.Bcast(wl, root=0)
		for i in range(1, n_procs):
			for j in range(len(job_lists[i])):
				j_full = job_lists[i][j]
				if rank == 0:
					mpi.COMM_WORLD.Send(data[j_full, :], i)
					mpi.COMM_WORLD.Send(var_noise[j_full, :], i)
				elif rank == i:
					mpi.COMM_WORLD.Recv(data[j, :], 0)
					mpi.COMM_WORLD.Recv(var_noise[j, :], 0)
		if rank == 0:
			root_jobs = [i in job_lists[0] for i in range(n_spectra)]
			data = data[root_jobs, :]
			var_noise = var_noise[root_jobs, :]
		mpi.COMM_WORLD.Bcast(full_data, root=0)

# if desired, perform a PCA of the input data and compress
# into space spanned by largest principal components
if precompress:

	# perform PCA
	import sklearn.decomposition as skd
	pca = skd.PCA()
	pca.fit(full_data)
	d_evals = pca.explained_variance_[::-1]
	ind_pc_sig = d_evals > \
				 np.max(d_evals) * eval_thresh
	n_pc_sig = np.sum(ind_pc_sig)
	proj_pc = pca.components_[n_pc_sig-1::-1, :]

	# project data and (inverse) noise covariances
	if rank == 0:
		print 'compressing: {:d} bins onto '.format(n_bins) + \
			  '{:d} principal components'.format(n_pc_sig)
	data = np.dot(data, proj_pc.T)
	full_data = np.dot(full_data, proj_pc.T)
	if datafile is None:
		inv_cov_noise = np.zeros((n_spectra, n_pc_sig, n_pc_sig))
		for i in job_lists[rank]:
			inv_cov_noise[i, :, :] = np.dot(proj_pc / \
											var_noise[i, :], \
											proj_pc.T)
		#inv_cov_noise = complete_array(inv_cov_noise, use_mpi)
		inv_cov_noise = complete_array_alt(inv_cov_noise, job_lists, \
										   use_mpi, last=False)
	else:
		n_jobs = len(job_lists[rank])
		inv_cov_noise = np.zeros((n_jobs, n_pc_sig, n_pc_sig))
		for i in range(n_jobs):
			inv_cov_noise[i, :, :] = np.dot(proj_pc / \
											var_noise[i, :], \
											proj_pc.T)
	n_bins_in = n_bins
	n_bins = n_pc_sig

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
	if datafile is None:
		mean_sample[:, k] = np.mean(data[in_class_k, :], 0)
		cov_sample[:, :, k] = np.cov(data[in_class_k, :], rowvar=False)
	else:
		mean_sample[:, k] = np.mean(full_data[in_class_k, :], 0)
		cov_sample[:, :, k] = np.cov(full_data[in_class_k, :], rowvar=False)
for j in range(n_spectra):
	#spectra_samples[j, :] = mean_sample[:, class_id_sample[j]]
	if datafile is None:
		spectra_samples[j, :] = data[j, :]
	else:
		spectra_samples[j, :] = full_data[j, :]
mean_samples = np.zeros((n_bins, n_classes, n_samples))
cov_samples = np.zeros((n_bins, n_bins, n_classes, n_samples))
class_probs_samples = np.zeros((n_classes, n_samples))
if n_classes > 1:
	class_id_samples = np.zeros((n_spectra, n_samples), dtype=int)
conds = np.zeros((n_classes, n_samples))
if datafile is not None:
	full_data = None
if recovery_test and rank == 0:
	rec_test_samples = np.zeros((n_bins, n_samples))
if save_spectra is not None and rank == 0:
	save_spectra_samples = np.zeros((n_save_spectra, n_bins, \
									 n_samples))

# Gibbs sample!
if sample:

	d_sample = n_samples / 10
	for i in range(n_samples):

		# report progress
		if np.mod(i, d_sample) == 0 and rank == 0:
			print i, '/', n_samples

		# invert current sample covariance and obtain determinant. 
		# use lapack directly for this to avoid two N^3 operations
		if not no_s_inv:
			inv_cov_sample = np.zeros((n_bins, n_bins, n_classes))
			ln_cov_det = np.zeros(n_classes)
			for k in range(n_classes):
				#inv_cov_sample[:, :, k] = npl.inv(cov_sample[:, :, k])
				chol_k = spl.dpotrf(cov_sample[:, :, k])[0]
				inv_cov_sample[:, :, k] = spl.dpotri(chol_k)[0]
				for j in range(n_bins):
					inv_cov_sample[j:, j, k] = inv_cov_sample[j, j:, k]
				ln_cov_det[k] = 2.0 * np.sum(np.log(np.diagonal(chol_k)))

		# sample multiple-class parameters
		if n_classes == 1:

			class_id_sample = np.zeros(n_spectra, dtype=int)

		else:

			# update categorical probabilities
			if rank == 0:
				n_in_class = np.array([np.sum(class_id_sample == k) for \
									   k in range(n_classes)])
				alpha = np.ones(n_classes) + n_in_class
				class_probs_sample = npr.dirichlet(alpha, 1)[0]
			else:
				class_probs_sample = np.zeros(n_classes)
			if use_mpi:
				mpi.COMM_WORLD.Bcast(class_probs_sample, root=0)

			# reassign classes
			class_id_sample = np.zeros(n_spectra, dtype=int)
			for j in job_lists[rank]:
				ln_class_probs_j = np.zeros(n_classes)
				for k in range(n_classes):
					delta = spectra_samples[j, :] - mean_sample[:, k]
					chisq = np.dot(delta, \
								   np.dot(inv_cov_sample[:, :, k], \
								   		  delta))
					ln_class_probs_j[k] = \
						-(chisq + ln_cov_det[k]) / 2.0 + \
						np.log(class_probs_sample[k])
				class_probs_j = np.exp(ln_class_probs_j - \
									   np.max(ln_class_probs_j))
				class_probs_j /= np.sum(class_probs_j)
				class_id_sample[j] = npr.choice(n_classes, \
												p=class_probs_j)
			class_id_sample = complete_array(class_id_sample, use_mpi)

		# calculate WF for each spectrum and use to draw true spectra
		spectra_samples = np.zeros((n_spectra, n_bins))
		for jj in range(len(job_lists[rank])):

			# ensure we get correct ids. j_full is the index into the 
			# full list of spectra, j is the index into the local 
			# data arrays. currently, these are only different in the 
			# apogee case
			j_full = job_lists[rank][jj]
			if datafile is None:
				j = j_full
			else:
				j = jj
			k = class_id_sample[j_full]

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

				# check if have spectrum-dependent noise, and whether 
				# noise is diagonal
				if inv_cov_noise is None:
					if len(var_noise.shape) == 2:
						inv_cov_noise_j = np.diag(1.0 / var_noise[j, :])
					else:
						inv_cov_noise_j = np.outer(mask[j, :] / \
												   var_noise, \
												   mask[j, :])
				else:
					if len(inv_cov_noise.shape) == 3:
						inv_cov_noise_j = inv_cov_noise[j, :, :]
					else:
						inv_cov_noise_j = np.outer(mask[j, :], \
												   mask[j, :]) * \
										  inv_cov_noise
				cov_wf = npl.inv(inv_cov_sample[:, :, k] + \
								 inv_cov_noise_j)
				cov_wf = symmetrize(cov_wf)
				mean_wf = np.dot(cov_wf, \
								 np.dot(inv_cov_sample[:, :, k], \
								 		mean_sample[:, k]) + \
								 np.dot(inv_cov_noise_j, data[j, :]))
				
			spectra_samples[j_full, :] = \
				npr.multivariate_normal(mean_wf, cov_wf, 1)
		spectra_samples = complete_array(spectra_samples, use_mpi)
		#test = complete_array_alt(spectra_samples, job_lists, use_mpi, \
		#						  last=False)

		# only class parameters require sampling; can do one class per 
		# mpi process. first sample means
		mean_sample = np.zeros((n_bins, n_classes))
		for k in class_job_lists[rank]:

			# class members
			in_class_k = (class_id_sample == k)
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
		for k in class_job_lists[rank]:

			# class members
			in_class_k = (class_id_sample == k)
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
		if n_classes > 1:
			class_probs_samples[:, i] = class_probs_sample
			class_id_samples[:, i] = class_id_sample
		for k in range(n_classes):
			conds[k, i] = npl.cond(cov_sample[:, :, k])
		if recovery_test and rank == 0:
			rec_test_samples[:, i] = spectra_samples[i_rec_test, :]
		if save_spectra is not None and rank == 0:
			for k in range(n_save_spectra):
				save_spectra_samples[k, :, i] = \
					spectra_samples[save_spectra_ids[k], :]

		# report if desired
		if diagnose and rank == 0:
			print 'sampled cov mat condition numbers:', conds[:, i]

	# store samples on disk
	if rank == 0:
		with h5py.File(io_base + 'samples.h5', 'w') as f:
			f.create_dataset('mean', data=mean_samples)
			f.create_dataset('covariance', data=cov_samples)
			if n_classes > 1:
				f.create_dataset('class_probs', \
								 data=class_probs_samples)
				f.create_dataset('class_id', \
								 data=class_id_samples)
			if recovery_test:
				f.create_dataset('rec_test', \
								 data=rec_test_samples)
		if save_spectra is not None:
			with h5py.File(io_base + 'save_spectra_samples.h5', \
						   'w') as f:
				f.create_dataset('save_spectra', \
								 data=save_spectra)
				f.create_dataset('save_spectra_samples', \
								 data=save_spectra_samples)

else:

	# retrieve samples
	if rank == 0:
		with h5py.File(io_base + 'samples.h5', 'r') as f:
			mean_samples = f['mean'][:]
			cov_samples = f['covariance'][:]
			n_bins, n_classes, n_samples = mean_samples.shape
			if n_classes > 1:
				class_probs_samples = f['class_probs'][:]
				class_id_samples = f['class_id'][:]
			if recovery_test:
				rec_test_samples = f['rec_test'][:]
			n_warmup = n_samples / 4
		if save_spectra is not None:
			with h5py.File(io_base + \
						   'save_spectra_samples.h5', 'r') as f:
				save_spectra = str(f['save_spectra'][...])
				save_spectra_samples = f['save_spectra_samples'][:]
				n_save_spectra = save_spectra_samples.shape[0]

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
		mp_perm_inv = list(np.argsort(mp_perm))
		print 'best permutation is', mp_perm
		print 'perms: ', perms
		print 'chi squares: ', chisq

	# determine class membership. number of samples of each id gives
	# the probability of class membership.
	# these classes have already been converted from sampled classes 
	# into true classes (if known)
	if rank == 0:
		p_class_ids = np.zeros((n_spectra, n_classes))
		map_class_id = np.zeros(n_spectra, dtype=int)
		for j in range(n_spectra):
			classes_sampled, class_counts = \
				np.unique(class_id_samples[j, :], return_counts=True)
			for i in range(len(classes_sampled)):
				if datafile is None:
					k = mp_perm[classes_sampled[i]]
				else:
					k = classes_sampled[i]
				p_class_ids[j, k] = float(class_counts[i]) / \
									float(n_samples)
			map_class_id[j] = np.argmax(p_class_ids[j, :])
		if datafile is None:
			correct_class = (map_class_id == class_ids)
			print '{:d}'.format(np.sum(correct_class)) + \
				  ' spectra correctly classified'

else:
	if datafile is None:
		mp_perm = [0]

# plots
if rank == 0:

	# quick recovery test plot
	if recovery_test:
		rec_test_mean = np.mean(rec_test_samples[:, n_warmup:], -1)
		rec_test_std = np.std(rec_test_samples[:, n_warmup:], -1)
		mp.fill_between(wl, rec_test_mean + rec_test_std, \
							rec_test_mean - rec_test_std, \
						color='LightGrey')
		mp.plot(wl, rec_test_data, 'k')
		mp.plot(wl, data[i_rec_test, :], 'r--')
		mp.axvline(wl[j_rec_test_lo], color='Grey', ls=':')
		mp.axvline(wl[j_rec_test_hi], color='Grey', ls=':')
		mp.xlabel(r'${\rm index}\,(i)$')
		mp.ylabel(r'${\rm flux}$')
		mp.savefig(io_base + 'recovery.pdf', bbox_inches='tight')
		mp.xlim(wl[j_rec_test_lo - 10], wl[j_rec_test_hi + 10])
		mp.savefig(io_base + 'recovery_zoom.pdf', bbox_inches='tight')
		mp.close()

	# summarize saved spectra
	if save_spectra is not None:
		fig, axes = mp.subplots(n_save_spectra, 1, \
								figsize=(16, 5 * n_save_spectra), \
								sharex=True)
		for i in range(n_save_spectra):
			#save_spectra_samples = np.zeros((n_save_spectra, n_bins, \
			#							 n_samples))
			d_mean = data[save_spectra_ids[i], :]
			d_std = np.sqrt(var_noise[save_spectra_ids[i], :])
			s_mean = np.mean(save_spectra_samples[i, :, n_warmup:], -1)
			s_std = np.std(save_spectra_samples[i, :, n_warmup:], -1)
			#axes[i].plot(wl, d_mean, 'r')
			#axes[i].plot(wl, s_mean, 'LightGrey')
			axes[i].fill_between(wl, d_mean + d_std, d_mean - d_std, \
								 color='r', alpha=0.5)
			axes[i].fill_between(wl, s_mean + s_std, s_mean - s_std, \
								 color='Grey', alpha=0.7)
			axes[i].set_xlabel(r'${\rm index}\,(i)$')
			axes[i].set_ylabel(r'${\rm flux}$')
			axes[i].set_xlim(wl[0], wl[-1])
			axes[i].set_ylim(0.5, 1.3)
			if i > 0:
				axes[i].set_yticklabels(axes[i].get_yticks())
				labels = axes[i].get_yticklabels()
				labels[-1] = ''
				axes[i].set_yticklabels(labels)
			if datafile is not None and window:
				y_pos = axes[i].get_ylim()[0]
				for j in range(n_windows):
					if j == 0:
						x_pos = wendices[j] / 2.0
					else:
						x_pos = (wendices[j] + wendices[j - 1]) / 2.0
					axes[i].axvline(wendices[j], color='k', lw=0.5, \
									ls=':')
					axes[i].text(x_pos, y_pos, wlabels[j], \
								 fontsize=8, ha='center', \
								 va='bottom')
		fig.subplots_adjust(hspace=0, wspace=0)
		mp.savefig(io_base + 'save_spectra.pdf', bbox_inches='tight')
		mp.close()

	# selection of trace plots
	fig, axes = mp.subplots(3, 1, figsize=(8, 5), sharex=True)
	for k in range(n_classes):
		axes[0].plot(mean_samples[0, k, :])
		axes[1].plot(mean_samples[n_bins / 2, k, :])
		axes[2].plot(mean_samples[-1, k, :])
	axes[2].set_xlabel('sample')
	axes[1].set_ylabel('mean')
	fig.subplots_adjust(hspace=0, wspace=0)
	mp.savefig(io_base + 'trace.pdf', bbox_inches='tight')
	mp.close()

	# class probability inference
	if n_classes > 1:
		fig, ax = mp.subplots(1, 1, figsize=(8, 5))
		cols = [cm(x) for x in np.linspace(0.1, 0.9, n_classes)]
		eval_bins = np.linspace(0.0, 1.0, 50)
		for k in range(n_classes):
			pretty_hist(class_probs_samples[k, :], eval_bins, ax, \
						cols[k])
			if datafile is None:
				ax.axvline(class_probs[k], color='k', ls='--')
		ax.set_xlabel(r'$\pi_i$')
		ax.set_ylabel(r'${\rm Pr}(\pi_i|d)$')
		mp.savefig(io_base + 'class_probs.pdf', bbox_inches='tight')
		mp.close()

	# class membership inference
	if n_classes > 1:
		fig, axes = mp.subplots(n_classes, 1, figsize=(5, 5 * n_classes))
		cols = [cm(x) for x in np.linspace(0.1, 0.9, 3)]
		eval_bins = np.linspace(0.0, 1.0, 50)
		for k in range(n_classes):
			pretty_hist(p_class_ids[:, k], eval_bins, axes[k], \
						cols[0], label='all')
			matches = (map_class_id == k)
			pretty_hist(p_class_ids[matches, k], eval_bins, axes[k], \
						cols[1], label='inferred class members')
			if datafile is None:
				matches = (class_ids == k)
				pretty_hist(p_class_ids[matches, k], eval_bins, axes[k], \
							cols[2], label='true class members')
			axes[k].set_xlabel(r'${\rm Pr(k=' + '{:d}'.format(k) + \
							   r')}$', fontsize=14)
			axes[k].set_ylabel(r'${\rm N[Pr(k=' + '{:d}'.format(k) + \
							   r')]}$', fontsize=14)
			axes[k].legend(loc='upper center', fontsize=12)
		mp.savefig(io_base + 'class_ids.pdf', bbox_inches='tight')
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
			res = mp_mean[:, mp_perm[k]] - mean[:, k]
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
			if window is not None:
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
		if window is not None:
			axes[-1].set_xlabel(r'${\rm index}\,(i)$')
		else:
			axes[-1].set_xlabel(r'$\lambda-15100\,[{\rm Angstroms}]$')
			mp.setp(axes[-1].get_xticklabels(), rotation=45)
	else:
		axes[-1].set_xlabel(r'index ($i$)', fontsize=14)
	fig.subplots_adjust(hspace=0, wspace=0)
	mp.savefig(io_base + 'mean.pdf', bbox_inches='tight')
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
			cax = axes[k, 1].matshow(mp_cov[:, :, mp_perm[k]], vmin=-ext_cov, \
									 vmax=ext_cov, cmap=mpcm.seismic, \
									 interpolation = 'nearest')
			axes[k, 2].matshow(mp_cov[:, :, mp_perm[k]] - cov[:, :, k], \
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
	mp.savefig(io_base + 'covariance.pdf', bbox_inches='tight')
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
			cax = axes[k, 1].matshow(mp_cor[:, :, mp_perm[k]], vmin=-1.0, \
									 vmax=1.0, cmap=mpcm.seismic, \
									 interpolation = 'nearest')
			axes[k, 2].matshow(mp_cor[:, :, mp_perm[k]] - cor[:, :, k], \
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
	mp.savefig(io_base + 'correlation.pdf', bbox_inches='tight')
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
		mp.savefig(io_base + 'gp_realizations.pdf', bbox_inches='tight')
		mp.close()

	# fun with ranks and eigenvalues of covariance matrices!
	n_eval_sig = np.zeros(n_classes, dtype=int)
	mp_cov_evals = np.zeros((n_bins, n_classes))
	mp_cov_evex = np.zeros((n_bins, n_bins, n_classes))
	fig, axes = mp.subplots(n_classes, 3, figsize=(16, 5 * n_classes))
	fig_e, axes_e = mp.subplots(1, 2, figsize=(16, 5))
	if n_classes == 1:
		axes = axis_to_axes(axes)
	if precompress:
		fig_p, axes_p = mp.subplots(n_classes, 3, figsize=(16, 5 * n_classes))
		if n_classes == 1:
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
			np.dot(proj_pc.T, \
				   np.dot(np.diag(d_evals[-n_pc_sig:]), \
						  proj_pc))
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
		for i in range(len(axes[k, :])):
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
	fig.savefig(io_base + 'low_rank_covariance.pdf', bbox_inches='tight')
	fig_e.savefig(io_base + 'evals.pdf', bbox_inches='tight')
	mp.close(fig)
	mp.close(fig_e)
	if precompress:
		fig_p.savefig(io_base + 'pca_vs_map.pdf', bbox_inches='tight')
		mp.close(fig_p)

	# condition numbers of sampled covariance matrices
	if sample:
		for k in range(n_classes):
			mp.semilogy(conds[k, :])
			if datafile is None:
				mp.axhline(npl.cond(cov[:, :, k]))
		mp.xlabel('index')
		mp.ylabel('condition number')
		mp.savefig(io_base + 'conds.pdf', bbox_inches='tight')
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
	mp.savefig(io_base + 'evex.pdf', bbox_inches='tight')
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
		fig.savefig(io_base + 'conditional_stddevs.pdf', bbox_inches='tight')
		mp.close(fig)
		fig_d.subplots_adjust(hspace=0)
		fig_d.savefig(io_base + 'conditional_inf_gain.pdf', bbox_inches='tight')
		mp.close(fig_d)
		fig_e.subplots_adjust(hspace=0)
		fig_e.savefig(io_base + 'most_correlated_stddevs.pdf', bbox_inches='tight')
		mp.close(fig_e)

		# additional plots if Cerium is included in the element list
		ce_windows = [ind for ind, label in enumerate(wlabels) \
					  if label == 'Ti']
		n_ce_windows = len(ce_windows)
		print n_ce_windows
		print ce_windows
		if n_ce_windows > 0:

			# plot predicted std devs (and covs?) for Cerium windows
			# given all other data
			fig, axes = mp.subplots(n_ce_windows, n_classes, \
									figsize=(8 * n_classes, \
											 5 * n_ce_windows), \
									sharex=True)
			if n_classes == 1:
				axes = axis_to_axes(axes, True)
			for k in range(n_classes):

				#inds_i = np.full(len(wl), False, dtype=bool)
				inds_o = np.full(len(wl), False, dtype=bool)
				for i in ce_windows:

					print wlabels[i]

					# now what? assume that all other bins observed (i.e., 
					# inds_i is everything other than this bin [and other 
					# Ce?])
					# better not to loop through windows but instead exclude
					# all Ce bins?

					# find indices
					if i == 0:
						inds_o = inds_o | (wl < wendices[i])
					else:
						inds_o = inds_o | ((wl >= wendices[i - 1]) & \
										   (wl < wendices[i]))
				inds_i = ~inds_o
				n_i = np.sum(inds_i)
				n_o = np.sum(inds_o)

				print n_i, n_o

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
				axes[k, 0].plot(wl[inds_o], np.sqrt(np.diag(cond_cov)))
				axes[k, 0].set_ylabel(r'$\sigma$')
			# zomg
		#what to plot? mean spectrum in those ranges?
		#conditional stddev from everything else?
		fig.subplots_adjust(hspace=0)
		fig.savefig(io_base + 'ce_conditional_stddevs.pdf', \
					bbox_inches='tight')
		mp.close(fig)





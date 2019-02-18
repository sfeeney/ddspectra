import numpy as np
import h5py
import matplotlib.pyplot as mp
import scipy.optimize as so

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

def vac2air(wave):
	''' https://github.com/jobovy/apogee/blob/master/apogee/tools/__init__.py '''
	x = (1.0e4 / wave) ** 2
	return wave / (1.0 + 0.05792105 / (238.0185 - x) + \
				   0.00167917 / (57.362 - x))

def air2vac(wave, alt=False):
	''' https://github.com/jobovy/apogee/blob/master/apogee/tools/__init__.py '''
	if alt:
		soln, conv = so.brentq(lambda x: vac2air_alt(x) - wave, \
							   wave - 20., wave + 20., \
							   full_output=True)
	else:
		soln, conv = so.brentq(lambda x: vac2air(x) - wave, \
							   wave - 20., wave + 20., \
							   full_output=True)
	if conv.converged:
		return soln
	else:
		return None

def vac2air_alt(l):
	s2 = (1.0e4 / l) ** 2
	n = 1 + 0.0000834254 + 0.02406147 / (130.0 - s2) + \
		0.00015998 / (38.9 - s2)
	return l / n

# settings
n_to_load = 30000
datafile = 'data/redclump_{:d}_alpha_nonorm.h5'
window = 'data/centers_final.txt' # 'data/centers_subset2_ce_nd.txt'
ce_lines_air = np.array([15784.75, 16376.48, 16595.18])
nd_lines_air = np.array([16053.628])

# convert lines from air to vacuum
ce_lines_vac = []
nd_lines_vac = []
for line in ce_lines_air:
	ce_lines_vac.append(air2vac(line))
	print 'Ce air', line, 'to vac', ce_lines_vac[-1]
for line in nd_lines_air:
	nd_lines_vac.append(air2vac(line))
	print 'Nd air', line, 'to vac', nd_lines_vac[-1]
ce_lines_vac = np.array(ce_lines_vac)
nd_lines_vac = np.array(nd_lines_vac)

# read data
wdata = np.genfromtxt(window, dtype=None, skip_header=1)
elements = [w[0] for w in wdata]
wdata_ce = wdata[elements.index('Ce')]
wdata_nd = wdata[elements.index('Nd')]

# Cerium/Neodymium targets: objects whose spectra are fully masked in 
# first Ce/Nd windows
i_tgt_ce = []
i_tgt_nd = []
n_loaded = 0
for n_file in range(1, 4):
	wl, full_data = read_spectra(n_to_load, \
								 datafile.format(n_file), \
								 return_wl=True)
	for i in range(1):

		# completely masked in first Ce window
		w_center = wdata_ce[i + 1]
		w_indices = (wl >= w_center - 2.5) & (wl <= w_center + 2.5)
		min_std = np.min(full_data[w_indices, :, 1], axis=0)
		i_tgt_ce += list(n_loaded + np.argwhere(min_std > 1000.0)[:, 0])

		# completely masked in first Nd window
		w_center = wdata_nd[i + 1]
		w_indices = (wl >= w_center - 2.5) & (wl <= w_center + 2.5)
		min_std = np.min(full_data[w_indices, :, 1], axis=0)
		i_tgt_nd += list(n_loaded + np.argwhere(min_std > 1000.0)[:, 0])

	n_loaded += full_data.shape[1]
np.savetxt('data/ids_ce_1_fully_masked.txt', i_tgt_ce, fmt='%d')
np.savetxt('data/ids_nd_1_fully_masked.txt', i_tgt_nd, fmt='%d')
np.savetxt('data/ids_ce_nd_1_fully_masked.txt', i_tgt_ce + i_tgt_nd, \
		   fmt='%d')

# low-SNR targets. read in SNRs, check ordering same as main datafile
# and store objects with lowest SNRs
n_save = 10
snr_ids = []
snrs = []
with open('data/snr_feeney.txt', 'r') as f:
	lines = f.readlines()
	for s in lines[1:]:
		data = s.split(' ')
		snr_ids.append(data[0])
		snrs.append(float(data[1].rstrip('\n')))
all_ids = []
for i in range(1, 4):
	with open('data/redclump_{:d}_ids.txt'.format(i), 'r') as f:
		lines = f.readlines()
		for s in lines:
			all_ids.append(s.split('.2-')[1].rstrip('.fits\n'))
if snr_ids == all_ids:
	print('SNR ordering identical to data ordering')
else:
	print('SNR ordering does not match data ordering')
	exit()
inds = range(len(snr_ids))
inds_sorted = [x for y, x in sorted(zip(snrs, inds))]
np.savetxt('data/ids_lowest_{:d}_snr.txt'.format(n_save), \
		   inds_sorted[0: n_save], fmt='%d')

# save all IDs for completeness
i_all = i_tgt_ce[0: n_save] + i_tgt_nd[0: n_save] + inds_sorted[0: n_save]
used = set()
i_all_unique = [x for x in i_all if x not in used and (used.add(x) or True)]
np.savetxt('data/ids_ce_nd_1_fully_masked_lowest_{:d}_snr.txt'.format(n_save), \
		   i_all_unique, fmt='%d')

ids = np.array([0, 1, 10, 16, 20, 21])
for i in range(len(ids)):
	print ids[i], all_ids[i_all_unique[ids[i]]], snrs[i_all_unique[ids[i]]]

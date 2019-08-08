import numpy as np
import matplotlib.pyplot as mp
mp.rc("font", family="serif", size=11.5)
mp.rc("text", usetex=True)
mp.rcParams['text.latex.preamble']=[r"\usepackage{bm}"]
import daft

# settings
use_class_prob_vec = True

# colours
s_color = {"ec": "#f89406"}
p_color = {"ec": "#46a546"}

# positions
data_level, like_level, obj_level, \
	obj_prior_level, pop_level, pop_prior_level, \
	pop_prior_par_level = np.arange(1, 8) - 0.25

# latex
def pr(vble):
	return r'{\rm P}(' + vble + ')'
def objectify(vble):
	return vble + r'_i'
alpha = r'\alpha'
spec_mean = r'\bm{m}_k'
spec_cov = r'\bm{S}_k'
pp_g = r'\beta' # r'\bm{\psi}^{\rm g}'
pp_c = r'\Omega' # r'\bm{\psi}^{\rm c}'
p_g = r'\theta' # r'\bm{\phi}^{\rm g}'
z = r'z'
c = r'\iota' # r'{\rm c}'
v = r'v'
class_prob = r'p_k'
class_prob_vec = r'\bm{p}'
spec_i = r'\bm{s}_i'
class_i = r'\kappa_i'
spec_mean_i = r'\bm{m}_{k=' + class_i + r'}'
spec_cov_i = r'\bm{S}_{k=' + class_i + r'}'
c_i = objectify(c)
v_i = objectify(v)
z_obs_i = r'\hat{z}_i'
v_obs_i = r'\hat{v}_i'
sig_z_obs_i = r'\sigma_{z,i}' # r'\sigma_{\hat{z}_i}'
sig_v_obs_i = r'\sigma_{v,i}' # r'\sigma_{\hat{v}_i}'
d_i = r'\hat{\bm{d}}_i'
noise_i = r'\bm{N}_i'

# create figure
pgm = daft.PGM([4.5, 7.25], origin=[0.0, 0.0], observed_style="inner")

# population-prior parameters
pgm.add_node(daft.Node('alpha', '$' + alpha + '$', 3.75, \
					   pop_prior_par_level, fixed=True))

# population priors
pgm.add_node(daft.Node('pr_spec_mean', '$' + pr(spec_mean) + '$', 0.75, pop_prior_level, \
					   aspect = 1.4, plot_params=s_color, \
					   shape='rectangle'))
pgm.add_node(daft.Node('pr_spec_cov', '$' + pr(spec_cov) + '$', 2.25, pop_prior_level, \
					   aspect = 1.4, plot_params=s_color, \
					   shape='rectangle'))
if use_class_prob_vec:
	pgm.add_node(daft.Node('pr_class_prob', \
						   '$' + pr(class_prob_vec + '|' + alpha) + '$', \
						   3.75, pop_prior_level, aspect = 1.6, \
						   plot_params=s_color, shape='rectangle'))
else:
	pgm.add_node(daft.Node('pr_class_prob', \
						   '$' + pr(class_prob + '|' + alpha) + '$', \
						   3.75, pop_prior_level, aspect = 1.6, \
						   plot_params=s_color, shape='rectangle'))

# population-level parameters
pgm.add_node(daft.Node('spec_mean', '$' + spec_mean + '$', 0.75, pop_level))
pgm.add_node(daft.Node('spec_cov', '$' + spec_cov + '$', 2.25, pop_level))
pgm.add_node(daft.Node('class_prob', '$' + class_prob + '$', 3.75, pop_level))

# object priors
pgm.add_node(daft.Node('pr_spec_i', \
					   '$' + pr(spec_i + '|' + spec_mean_i + ',' + \
								spec_cov_i) + '$', \
					   2.25, obj_prior_level, aspect = 3.6, \
					   plot_params=s_color, shape='rectangle'))
if use_class_prob_vec:
	pgm.add_node(daft.Node('pr_class_i', \
						   '$' + pr(class_i + '|' + class_prob_vec) + '$', \
						   3.75, obj_prior_level, aspect = 1.6, \
						   plot_params=s_color, shape='rectangle'))
else:
	pgm.add_node(daft.Node('pr_class_i', \
						   '$' + pr(class_i + '|' + class_prob) + '$', \
						   3.75, obj_prior_level, aspect = 1.6, \
						   plot_params=s_color, shape='rectangle'))

# object-level parameters
pgm.add_node(daft.Node('noise_i', '$' + noise_i + '$', 0.75, obj_level, \
					   fixed=True))
pgm.add_node(daft.Node('spec_i', '$' + spec_i + '$', 2.25, obj_level))
pgm.add_node(daft.Node('class_i', '$' + class_i + '$', 3.75, obj_level))

# likelihoods
pgm.add_node(daft.Node('pr_d_i', \
					   '$' + pr(d_i + '|' + spec_i + ',' + noise_i) + '$', \
					   2.25, like_level, aspect=2.6, plot_params=s_color, \
					   shape='rectangle'))

# observables
pgm.add_node(daft.Node('d_i', '$' + d_i + '$', 2.25, data_level, \
					   observed=True))

# edges
pgm.add_edge('alpha', 'pr_class_prob')
pgm.add_edge('pr_spec_mean', 'spec_mean')
pgm.add_edge('pr_spec_cov', 'spec_cov')
pgm.add_edge('pr_class_prob', 'class_prob')
pgm.add_edge('spec_mean', 'pr_spec_i')
pgm.add_edge('spec_cov', 'pr_spec_i')
pgm.add_edge('class_prob', 'pr_class_i')
pgm.add_edge('class_i', 'pr_spec_i')
pgm.add_edge('pr_spec_i', 'spec_i')
pgm.add_edge('pr_class_i', 'class_i')
pgm.add_edge('noise_i', 'pr_d_i')
pgm.add_edge('spec_i', 'pr_d_i')
pgm.add_edge('pr_d_i', 'd_i')

# object plate
pgm.add_plate(daft.Plate([0.25, data_level - 0.5, 4.0, \
						  pop_level - data_level - 0.05], \
						  label=r"$1 \le i \le n_{\rm s}$", \
						  shift=-0.0, rect_params={"ec": "r"}, \
						  label_offset=(2, 2)))
pgm.add_plate(daft.Plate([0.25, pop_level - 0.5, 4.0, \
						  pop_prior_par_level - pop_level], \
						  label=r"$1 \le k \le n_{\rm c}$", \
						  shift=-0.0, rect_params={"ec": "r"}, \
						  label_offset=(2, 2)))

# render and save
pgm.render()
pgm.figure.savefig('bhm_plot.pdf')

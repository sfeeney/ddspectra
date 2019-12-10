# SSSpaNG: Stellar Spectra as Sparse, data-driven Non-Gaussian processes

![](http://45.media.tumblr.com/tumblr_m7bsi0W4Wz1qb05aco1_500.gif)

(copyright BBC, I would expect...)

SSSpaNG is a data-driven Gaussian Process model of the spectra of APOGEE red clump stars, whose parameters we infer using Gibbs sampling. By pooling information between stars to infer their covariance we permit clear identification of the correlations between spectral pixels. Harnessing this correlation structure, we infer a complete spectrum for each red clump star, inpainting missing regions and de-noising by a factor of at least 2-3 for low-signal-to-noise stars. These high-fidelity inferred spectra will enable improved measurements for a full set of elemental abundances for each star. Our model also allows us to quantify the information gained by observing portions of a star’s spectrum, and thereby define the most mutually informative spectral regions. Using 25 windows centred on elemental absorption lines, we demonstrate that the iron-peak and alpha-process elements are particularly mutually informative for these spectra, and that the majority of information about a target window is contained in the 10-or-so most informative windows. Our information-gain metric has the potential to inform models of nucleosynthetic yields and optimize the design of future observations.

The two main (Python) codes required to run SSSpaNG (nb_conversion.py) and process the results (window_results.py) require the numpy, scipy, itertools, h5py, matplotlib, mpi4py and sklearn modules.

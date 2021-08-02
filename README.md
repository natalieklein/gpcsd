# GPCSD (Gaussian process current source density estimation)

Python implementation of Gaussian process current source density (CSD) estimation.

Paper: Klein, N., Siegle, J.H., Teichert, T., Kass, R.E. (2021) Cross-population coupling of neural activity based on Gaussian process current source densities. (preprint: https://arxiv.org/abs/2104.10070)

This code estimates current source density over space and time from local field potential (LFP) recordings.

Full source code available at https://github.com/natalieklein/gpcsd.

## Installation
Install using `pip install gpcsd`. 
To get the scripts to reproduce the paper results, get the source code from github: https://github.com/natalieklein/gpcsd.
Dependencies may be installed as described below.

## Dependencies
The file `environment.yml` in the full source code can be used to set up an anaconda environment called `gpcsd` with most of the required packages.
These include very common packages (`numpy`, `scipy`, `matplotlib`, `h5py`, `ipykernel`) along with a few more specialized packages (`scikit-image`, `joblib`, `networkx`, `tqdm`, and `autograd`). 
You will also need to install the `kCSD-python` package manually (download source from https://github.com/Neuroinflab/kCSD-python/releases/tag/v2.0, then `cd` to the `kCSD-python/` directory and run `python setup.py install` inside activated anaconda environment).
Finally, there are some Matlab files for the torus graphs analysis. 
Using Matlab to run these files is the fastest option, but you may also be able to use Octave with appropriate setup.
See https://github.com/natalieklein/torus-graphs/tree/octave for more information.
You will need to download the torus graphs code from https://github.com/natalieklein/torus-graphs and add it to your Matlab/Octave path to run the Matlab scripts.

## Main source code
Directory `src/gpcsd` contains the main source code. There are classes `gpcsd1d.py` and `gpcsd2d.py` for the GPCSD models, in addition to some support functions in other files.

## Simulation studies
Simulation studies are found in `simulation_studies` and reproduce all simulation results shown in the paper. See `simulation_studies/README.md` for a full description of the scripts.

## Auditory LFP analysis
Here we apply GPCSD1D to two-probe auditory cortex LFPs measured in a macaque monkey. The scripts reproduce all results shown in the paper. See `auditory_lfp/README.md` for more information on the scripts.
The auditory LFP data can be downloaded from https://doi.org/10.5281/zenodo.5137888. 
The code assumes that it will be downloaded into `auditory_lfp/data/`. 

## Neuropixels analysis
We apply GPCSD2D to LFP recordings from Neuropixels probes in a mouse. This reproduces the figures shown in the paper. See `neuropixels/README.md` for more detail.
The Neuropixels data can be downloaded from https://doi.org/10.5281/zenodo.5150708.
The code assumes that it will be downloaded into `neuropixels/data/`.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4698746.svg)](https://doi.org/10.5281/zenodo.4698746)




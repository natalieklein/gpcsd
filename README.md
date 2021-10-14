# GPCSD (Gaussian process current source density estimation)

Python implementation of Gaussian process current source density (CSD) estimation.

Paper: Klein, N., Siegle, J.H., Teichert, T., Kass, R.E. (2021) Cross-population coupling of neural activity based on Gaussian process current source densities. (preprint: https://arxiv.org/abs/2104.10070)

This code estimates current source density over space and time from local field potential (LFP) recordings.

Full source code available at https://github.com/natalieklein/gpcsd.

## Installation
Install using `pip install gpcsd`. 
To get the scripts to reproduce the paper results, get the source code from github: https://github.com/natalieklein/gpcsd.
Dependencies should be automatically installed by `pip`, but also see the Anaconda `environment.yml` file.
It will install two github-based dependencies with `pip` that are not able to be included in `setup.py`.

## Main source code
Directory `src/gpcsd` contains the main source code. There are classes `gpcsd1d.py` and `gpcsd2d.py` for the GPCSD models, in addition to some support functions in other files.

## Simulation studies
Simulation studies are found in `simulation_studies` and reproduce all simulation results shown in the paper. See `simulation_studies/README.md` for a full description of the scripts.

## Auditory LFP analysis
Here we apply GPCSD1D to two-probe auditory cortex LFPs measured in a macaque monkey. The scripts reproduce all results shown in the paper. See `auditory_lfp/README.md` for more information on the scripts.
The auditory LFP data can be downloaded from https://doi.org/10.5281/zenodo.5137888, or using script `download_data.sh`.
The code assumes that it will be downloaded into `auditory_lfp/data/`. 

## Neuropixels analysis
We apply GPCSD2D to LFP recordings from Neuropixels probes in a mouse. This reproduces the figures shown in the paper. See `neuropixels/README.md` for more detail.
The Neuropixels data can be downloaded from https://doi.org/10.5281/zenodo.5150708, or using script `download_data.sh`.
The code assumes that it will be downloaded into `neuropixels/data/`.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5154196.svg)](https://doi.org/10.5281/zenodo.5154196)





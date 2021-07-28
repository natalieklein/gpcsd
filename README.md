# GPCSD (Gaussian process current source density estimation)

Python implementation of Gaussian process current source density (CSD) estimation.

Paper: Klein, N., Siegle, J.H., Teichert, T., Kass, R.E. (2021) Cross-population coupling of neural activity based on Gaussian process current source densities. (preprint: https://arxiv.org/abs/2104.10070)

This code estimates current source density over space and time from local field potential (LFP) recordings.

## Main source code
Directory `src/gpcsd` contains the main source code. There are classes `gpcsd1d.py` and `gpcsd2d.py` for the GPCSD models, in addition to some support functions in other files.

## Simulation studies
Simulation studies are found in `simulation_studies` and reproduce all simulation results shown in the paper. See `simulation_studies/README.md` for a full description of the scripts.

## Auditory LFP analysis
Here we apply GPCSD1D to two-probe auditory cortex LFPs measured in a macaque monkey. The scripts reproduce all results shown in the paper. See `auditory_lfp/README.md` for more information on the scripts and how to download the data.

## Neuropixels analysis
We apply GPCSD2D to LFP recordings from Neuropixels probes in a mouse. This reproduces the figures shown in the paper. See `neuropixels/README.md` for more detail.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4698746.svg)](https://doi.org/10.5281/zenodo.4698746)




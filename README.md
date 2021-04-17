# gpcsd (Gaussian process current source density estimation)

Python implementation of Gaussian process current source density (CSD) estimation.

Paper: Klein, N., Siegle, J.H., Teichert, T., Kass, R.E. (2021) Cross-population coupling of neural activity based on Gaussian process current source densities. (bioRxiv preprint -- coming soon)

This code estimates current source density over space and time from local field potential (LFP) recordings.

## 1D 
1D LFP recordings include those made by laminar/linear probes. The forward model assumes a cylinder of radius $R$ around the probe with constant CSD value.
`demo_1D.py` generates ground truth data, then compares traditional second-spatial-derivative-based CSD estimation to GPCSD estimation.

## 2D
We consider 2D LFP recordings from Neuropixels probes and make certain assumptions in the forward model for this case. The forward model could be modified slightly to be appropriate for other 2D recording scenarios (such as Utah arrays).
`demo_2D.py` generates ground truth data and again compares traditional CSD estimation to GPCSD estimation.

## More info
The demos above use the fixed, true Gaussian process hyperparameters to demonstrate the method. In practice, the Gaussian process marginal likelihood (found in `gp_lik.py`) can be optimized with respect to the hyperparameters using standard optimization libraries.


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4698746.svg)](https://doi.org/10.5281/zenodo.4698746)




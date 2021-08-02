## Simulation studies 

Simulation studies from the paper "Cross-population coupling of neural activity based on Gaussian process current source densities" by Klein, N., Siegle, J.H, Teichert, T., and Kass, R.E. (preprint: https://arxiv.org/abs/2104.10070).

### Description of scripts
* simple_template_1D.py: generates a single trial of a simple dipole-like CSD pattern, forward models LFP with or without white noise, and compares the results of traditional CSD, GPCSD and kCSD. Produces Figure 1 in the paper.
* sim_from_gp_1D.py: generates multiple trials from a GPCSD model and compares results of traditional CSD, GPCSD, and kCSD. In this case, the fitted GPCSD model form matches the generating form. Produces numerical results from the paper.
* sim_from_gp_1D_mismatch.py: generates multiple trials from a GPCSD model and compares results of traditional CSD, GPCSD, and kCSD. In this case, the fitted GPCSD model form does not match the generating model form (two different examples). Produces numerical results from the paper Supplement.
* sim_from_gp_2D.py: generates data from a GPCSD2D model and tests hyperparameter optimization.

## Simulation studies 

### Description of scripts
* simple_template_1D.py: generates a single trial of a simple dipole-like CSD pattern, forward models LFP with or without white noise, and compares the results of traditional CSD, GPCSD and kCSD. Produces Figure 1 in the paper.
* sim_from_gp_1D.py: generates multiple trials from a GPCSD model and compares results of traditional CSD, GPCSD, and kCSD. In this case, the fitted GPCSD model form matches the generating form.
* sim_from_gp_1D_mismatch.py: generates multiple trials from a GPCSD model and compares results of traditional CSD, GPCSD, and kCSD. In this case, the fitted GPCSD model form does not match the generating model form (two different examples).

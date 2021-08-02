## Neuropixels data analysis

### Assumed directory structure
The scripts assume the data is stored in `data/` and results will be saved in `results/`.

### Data download
Data can be downloaded from https://doi.org/10.5281/zenodo.5150708.

### Software
Most of the analysis is done in Python using the scripts here.
The exception is the torus graph analysis, which is done in Matlab.
It could possibly be run in Octave if using the correct branch from
https://github.com/natalieklein/torus-graphs, but it is noticeably slower.

### Description of scripts
* `extract_data.py`: extracts data from `nwb` files to get the visual area LFPs of interest; produces Figure 6.A from the paper.
* `fit_gpcsd2d.py`: fits GPSD2D model to extracted data, does filtering to extract phases for torus graph analysis.
* `fit_torus_graph.m`: does torus graph fitting and bootstrapping on phases saved by `fit_gpcsd2d.py`.
* `viz_torus_graph.py`: loads results saved by `fit_torus_graph.m` to produce Figure 6.B/C in the paper.


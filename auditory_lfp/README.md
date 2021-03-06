## Auditory data analysis

### Assumed directory structure
The scripts assume the data is stored in `data/` and results will be saved in `results/`.

### Data download
Data can be downloaded from https://doi.org/10.5281/zenodo.5137888, or using the script `download_data.sh` in the project root directory.

### Description of scripts
* `fit_gpcsd_baseline.py`: fits GPCSD1D model to baseline period of trial for both probes, computes oscillatory power and phase and produces Figure 2 in the paper.
* `torus_graph_fit.py`: loads phases saved from `fit_gpcsd_baseline.py` to do torus graph fitting and bootstrapping.
* `viz_torus_graph.py`: loads torus graph parameters saved from `torus_graph_fit.py` to visualize; produces Figure 3 in the paper.
* `fit_mean_function.py`: estimates mean evoked CSD using GPCSD1D model from baseline period, segments CSD evoked into components, estimates per-trial shifts for each component, and does correlation analysis on shifts. Produces Figures 4 and 5 in the paper.

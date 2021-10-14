"""
Fit torus graph to auditory phase data using pyTG.

"""

# %% Imports
import numpy as np
from tqdm import tqdm
import os.path
root_path = '/'.join(os.path.abspath(__file__).split('/')[:-1])

from pyTG import torusGraphs
from scipy.io import loadmat, savemat

# indicate whether to refit model/redo bootstrap if results files exist
refit_model = False
redo_bootstrap = False
nboot = 2 # warning: can take a long time; use small nboot for testing, 100 used in paper

# %% Each phase entry has 'csd' and 'lfp' keys
phases = {}
for probe in ['lateral', 'medial']:
    phases[probe] = loadmat('%s/results/csd_lfp_filt_phases_%s.mat' % (root_path, probe))

# %% Fit phase differences submodel
if refit_model or not os.path.isfile('%s/results/twoprobe_tg.mat' % (root_path)):
    submodels = (False, True, False)

    print('starting CSD torus graph fit')
    X = np.vstack([phases['lateral']['csd'], phases['medial']['csd']])
    graph, _, _, nodepairs, _, phi, phi_cov = torusGraphs(X, selMode=submodels)
    phi_hat_csd = phi
    csd_pvals = nodepairs['pVals']

    print('starting LFP torus graph fit')
    X = np.vstack([phases['lateral']['lfp'], phases['medial']['lfp']])
    graph, _, _, nodepairs, _, phi, phi_cov = torusGraphs(X, selMode=submodels)
    phi_hat_lfp = phi
    lfp_pvals = nodepairs['pVals']

    # Save results
    dict = {'twoprobe_csd_pvals': csd_pvals.squeeze(), 'twoprobe_lfp_pvals': lfp_pvals.squeeze(), 
            'phi_hat_lfp': phi_hat_lfp.squeeze(), 'phi_hat_csd': phi_hat_csd.squeeze()}
    savemat('%s/results/twoprobe_tg.mat' % root_path, dict)
    print('Twoprobe CSD TG has %d edges bonf 0.001' % np.sum(csd_pvals < 0.001/(24*24)))
    print('Twoprobe LFP TG has %d edges bonf 0.001' % np.sum(lfp_pvals < 0.001/(24*24)))

# %% Bootstrapping 
if redo_bootstrap or not os.path.isfile('%s/results/bootstrap_pplv_tg.mat' % root_path):
    ntrials = phases['lateral']['csd'].shape[1]
    partial_plv_csd = np.zeros((1128, nboot));
    for bi in tqdm(range(nboot), desc='Bootstrap iteration'):
        binds = np.random.choice(ntrials, size=ntrials, replace=True)
        X = np.vstack([phases['lateral']['csd'][:, binds], phases['medial']['csd'][:, binds]])
        graph, _, _, nodepairs, _, phi, phi_cov = torusGraphs(X, selMode=submodels)
        partial_plv_csd[:, bi] = nodepairs['condCoupling']
    # Save results
    savemat('%s/results/bootstrap_pplv_tg.mat' % root_path, {'partial_plv_csd': partial_plv_csd})


"""
Fit torus graph to auditory phase data using pyTG.

"""

# %% Imports
import numpy as np

from pyTG import torusGraphs
from scipy.io import loadmat, savemat

# %% Each phase entry has 'csd' and 'lfp' keys
phases = {}
phases['lateral'] = loadmat('results/csd_lfp_filt_phases_lateral.mat')
phases['medial'] = loadmat('results/csd_lfp_filt_phases_lateral.mat')

# %% Fit phase differences submodel
submodels = [False, True, False]

X = np.vstack([phases['lateral']['csd'], phases['medial']['csd']])
graph, _, _, nodepairs, _, phi, phi_cov = torusGraphs(X, selMode=submodels)
phi_hat_csd = phi
csd_pvals = nodepairs['pVals']

X = np.vstack([phases['lateral']['lfp'], phases['medial']['lfp']])
graph, _, _, nodepairs, _, phi, phi_cov = torusGraphs(X, selMode=submodels)
phi_hat_lfp = phi
lfp_pvals = nodepairs['pVals']

# %% Save results
dict = {'twoprobe_csd_pvals': csd_pvals.squeeze(), 'twoprobe_lfp_pvals': lfp_pvals.squeeze(), 
        'phi_hat_lfp': phi_hat_lfp.squeeze(), 'phi_hat_csd': phi_hat_csd.squeeze()}
savemat('results/twoprobe_tg.mat', dict)

print('Twoprobe CSD TG has %d edges bonf 0.001' % np.sum(csd_pvals < 0.001/(24*24)))
print('Twoprobe LFP TG has %d edges bonf 0.001' % np.sum(lfp_pvals < 0.001/(24*24)))

# %% Bootstrapping (TODO speed up pyTG first) for now copied old results to results/


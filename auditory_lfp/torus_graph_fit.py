"""
Fit torus graph to auditory phase data using pyTG.

"""

# %% Imports
import numpy as np
from tqdm import tqdm

from pyTG import torusGraphs
from scipy.io import loadmat, savemat

# %% Each phase entry has 'csd' and 'lfp' keys
phases = {}
for probe in ['lateral', 'medial']:
        phases[probe] = loadmat('results/csd_lfp_filt_phases_%s.mat' % probe)

# %% Fit phase differences submodel
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

# %% Save results
dict = {'twoprobe_csd_pvals': csd_pvals.squeeze(), 'twoprobe_lfp_pvals': lfp_pvals.squeeze(), 
        'phi_hat_lfp': phi_hat_lfp.squeeze(), 'phi_hat_csd': phi_hat_csd.squeeze()}
savemat('results/twoprobe_tg.mat', dict)

print('Twoprobe CSD TG has %d edges bonf 0.001' % np.sum(csd_pvals < 0.001/(24*24)))
print('Twoprobe LFP TG has %d edges bonf 0.001' % np.sum(lfp_pvals < 0.001/(24*24)))

# %% Bootstrapping 
nboot = 100 # warning: can take a long time; use small nboot for testing, 100 used in paper
ntrials = phases['lateral']['csd'].shape[1]
partial_plv_csd = np.zeros((1128, nboot));
for bi in tqdm(range(nboot), desc='Bootstrap iteration'):
     binds = np.random.choice(ntrials, size=ntrials, replace=True)
     X = np.vstack([phases['lateral']['csd'][:, binds], phases['medial']['csd'][:, binds]])
     graph, _, _, nodepairs, _, phi, phi_cov = torusGraphs(X, selMode=submodels)
     partial_plv_csd[:, bi] = nodepairs['condCoupling']
# Save results
savemat('results/bootstrap_pplv_tg.mat', {'partial_plv_csd': partial_plv_csd})


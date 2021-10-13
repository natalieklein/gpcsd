"""
Fit torus graph to neuropixels phase data using pyTG.
"""

# 'results/neuropixel_csd_V1_phases.mat' (or LM)
# each has beta and theta for two different time points
# so fit model for each frequency/time, also bootstrap. 
# Should be similar in structure to the 1D script. 

# %% Imports
import numpy as np
from tqdm import tqdm

from pyTG import torusGraphs
from scipy.io import loadmat, savemat

# %% Each area has 'beta' and 'theta' keyed phases
# shape: i.e. phases['V1']['theta']['shape'] is (4, 2, 129) - (locations, time points, trials)
phases = {}
for area in ['V1', 'LM']:
    phases[area] = loadmat('results/neuropixel_csd_%s_phases.mat' % area)
times = [0, 70] # ms

# %% Fit phase differences submodels
submodels = (False, True, False)

phi_hat = {'theta': {}, 'beta': {}}
pvals = {'theta': {}, 'beta': {}}
for i, t in enumerate(times):
    for band in ['theta', 'beta']:
        print('starting TG fit for t=%dms, %s band' % (t, band))
        X = np.vstack([phases['V1'][band][:, i, :], phases['LM'][band][:, i, :]])

        graph, _, _, nodepairs, _, phi, phi_cov = torusGraphs(X, selMode=submodels)
        phi_hat[band][t] = phi
        pvals[band][t] = nodepairs['pVals']
        print('%s t=%d TG has %d edges 0.001' % (band, t, np.sum(pvals[band][t] < 0.001)))

# %% save results
res = {'theta_pvals': np.stack([pvals['theta'][0], pvals['theta'][70]], axis=1),
       'theta_phi': np.stack([phi_hat['theta'][0], phi_hat['theta'][70]], axis=1),
       'beta_pvals': np.stack([pvals['beta'][0], pvals['beta'][70]], axis=1),
       'beta_phi': np.stack([phi_hat['beta'][0], phi_hat['beta'][70]], axis=1),
      }
savemat('results/neuropixel_tg.mat', res)

# %% Bootstrapping 
nboot = 1000
ntrials = phases['V1']['theta'].shape[2]
partial_plv = {'theta': {0:[], 70:[]}, 'beta': {0:[], 70:[]}}
for bi in tqdm(range(nboot), desc='Bootstrap iteration'):
    binds = np.random.choice(ntrials, size=ntrials, replace=True)
    for i, t in enumerate(times):
        for band in ['theta', 'beta']:
            X = np.vstack([phases['V1'][band][:, i, binds], phases['LM'][band][:, i, binds]])

            graph, _, _, nodepairs, _, phi, phi_cov = torusGraphs(X, selMode=submodels)

            partial_plv[band][t].append(nodepairs['condCoupling'])
   
# Save results TODO
pplv_theta = np.stack([np.array(partial_plv['theta'][0]), np.array(partial_plv['theta'][70])])
pplv_beta = np.stack([np.array(partial_plv['beta'][0]), np.array(partial_plv['beta'][70])])

res = {'pplv_theta': pplv_theta, 'pplv_beta': pplv_beta}
savemat('results/bootstrap_neuropixels.mat', res)


# %%

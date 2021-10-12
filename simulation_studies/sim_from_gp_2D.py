"""
Simulate some data from 2D GP to test methods and evaluate errors.

"""

# %% Imports
import matplotlib.pyplot as plt
import autograd.numpy as np

from gpcsd.gpcsd2d import GPCSD2D
from gpcsd.covariances import *
from gpcsd.utility_functions import normalize
from gpcsd.predict_csd import predictcsd_trad_2d

np.random.seed(1)

 # %%
# Setup
ntrials = 1

a1 = 0
b1 = 60
a2 = 0
b2 = 1500
nt = 10
t = np.linspace(0, 100, nt)[:, None]
ngl1 = 30
ngl2 = 80

nx1 = 4
nx2 = 50
x1 = np.linspace(a1, b1, nx1)[:, None]
x2 = np.linspace(a2, b2, nx2)[:, None]
x_grid = expand_grid(x1, x2)
nz1 = 20
nz2 = 500
z1 = np.linspace(a1, b1, nz1)[:, None]
z2 = np.linspace(a2, b2, nz2)[:, None]
z_grid = expand_grid(z1, z2)

R_true = 30.0
ellSE1_true = 40.0
ellSE2_true = 100.0
sig2tM_true = 10
elltM_true = 1
sig2tSE_true = 20
elltSE_true = 5
sig2n_true = 0.5
eps = 10.0

gpcsd_gen = GPCSD2D(np.zeros((z_grid.shape[0], nt, ntrials)), x=z_grid, t=t, 
                    a1=a1, b1=b1, a2=a2, b2=b2, 
                    temporal_cov_list=[GPCSDTemporalCovSE(t), GPCSDTemporalCovMatern(t)], 
                    ngl1=ngl1, ngl2=ngl2, eps=eps)
gpcsd_gen.R['value'] = R_true
gpcsd_gen.sig2n['value'] = sig2n_true
gpcsd_gen.spatial_cov.params['ell1']['value'] = ellSE1_true
gpcsd_gen.spatial_cov.params['ell2']['value'] = ellSE2_true
gpcsd_gen.temporal_cov_list[0].params['ell']['value'] = elltSE_true
gpcsd_gen.temporal_cov_list[0].params['sigma2']['value'] = sig2tSE_true
gpcsd_gen.temporal_cov_list[1].params['ell']['value'] = elltM_true
gpcsd_gen.temporal_cov_list[1].params['sigma2']['value'] = sig2tM_true

# %% Generate CSD on dense spatial grid
csd_dense, _ = gpcsd_gen.sample_prior(1, type="csd")
csd_dense_rect = csd_dense.reshape((nz1, nz2, nt, -1))

# %% Pass through forward model to get LFP at sparse spatial grid
lfp_sparse = np.atleast_3d(fwd_model_2d(csd_dense_rect, z1, z2, x_grid, R_true, gpcsd_gen.eps)) 
lfp_sparse += np.random.normal(0, np.sqrt(sig2n_true), lfp_sparse.shape)
lfp_sparse_rect = lfp_sparse.reshape((nx1, nx2, nt, -1))

#%% Visualize
plt.figure(figsize=(14, 10))
for ti in [0,1,2,3]:
    plt.subplot(2,4,ti+1)
    plt.imshow(lfp_sparse_rect[:, :, ti, 0].T, aspect='auto', cmap='bwr',
               vmin=-np.nanmax(np.abs(lfp_sparse_rect[:, :, ti, 0])), 
               vmax=np.nanmax(np.abs(lfp_sparse_rect[:, :, ti, 0])))
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('LFP (t = %0.2f)' % t[ti])
for ti in [0,1,2,3]:
    plt.subplot(2,4,ti+1+4)
    plt.imshow(csd_dense_rect[:, :, ti, 0].T, aspect='auto', cmap='bwr',
               vmin=-np.nanmax(np.abs(csd_dense_rect[:, :, ti, 0])), 
               vmax=np.nanmax(np.abs(csd_dense_rect[:, :, ti, 0])))
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('CSD (t = %0.2f)' % t[ti])
plt.tight_layout()

# %% Predict CSD and LFP from generative model with correct parameters
gpcsd_gen.update_lfp(lfp_sparse, t, x_grid)
gpcsd_gen.predict(z_grid, t, type="csd")
gpcsd_gen.predict(x_grid, t, type="lfp")
csd_pred_rect = gpcsd_gen.csd_pred.reshape((nz1, nz2, nt, -1))
lfp_pred_rect = gpcsd_gen.lfp_pred.reshape((nx1, nx2, nt, -1))

plt.figure(figsize=(14, 10))
for ti in [0,1,2,3]:
    plt.subplot(2,4,ti+1)
    plt.imshow(lfp_pred_rect[:, :, ti, 0].T, aspect='auto', cmap='bwr',
               vmin=-np.nanmax(np.abs(lfp_sparse_rect[:, :, ti, 0])), 
               vmax=np.nanmax(np.abs(lfp_sparse_rect[:, :, ti, 0])))
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Pred LFP (t = %0.2f)' % t[ti])
for ti in [0,1,2,3]:
    plt.subplot(2,4,ti+1+4)
    plt.imshow(csd_pred_rect[:, :, ti, 0].T, aspect='auto', cmap='bwr',
               vmin=-np.nanmax(np.abs(csd_dense_rect[:, :, ti, 0])), 
               vmax=np.nanmax(np.abs(csd_dense_rect[:, :, ti, 0])))
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Pred CSD (t = %0.2f)' % t[ti])
plt.tight_layout()

# %% Set up GPCSD model
R_prior = GPCSDInvGammaPrior()
R_prior.set_params(10, 200)
ellSEprior = GPCSDInvGammaPrior()
ellSEprior.set_params(10, 20)
temporal_cov_SE = GPCSDTemporalCovSE(t, ell_prior=ellSEprior)
ellMprior = GPCSDInvGammaPrior()
ellMprior.set_params(1, 5)
temporal_cov_M = GPCSDTemporalCovMatern(t, ell_prior=ellMprior)
gpcsd_model = GPCSD2D(lfp_sparse, x_grid, t, a1, b1, a2, b2, ngl1, ngl2, R_prior=R_prior,
                      eps=gpcsd_gen.eps, temporal_cov_list=[temporal_cov_SE, temporal_cov_M])
print(gpcsd_model)

# %% Fit GPCSD model
gpcsd_model.fit(n_restarts=10, verbose=True)

# %% predict
print(gpcsd_model)
gpcsd_model.predict(z_grid, t)
csd_pred_rect = gpcsd_model.csd_pred.reshape((nz1, nz2, nt, -1))

# %% trad CSD on test data
tcsd_pred = predictcsd_trad_2d(lfp_sparse_rect)[:, 1:-1, :, :]

# %%
tinds = [0, 1, 2, 3]
plt.figure(figsize=(16, 16))
for ti in tinds:
    plt.subplot(4,4,ti+1)
    plt.imshow(normalize(tcsd_pred[:, :, ti, 0]).T, aspect='auto', cmap='bwr')

for ti in tinds:
    plt.subplot(4,4,ti+1+4)
    plt.imshow(normalize(csd_pred_rect[:, :, ti, 0]).T, aspect='auto', cmap='bwr')

for ti in tinds:
    plt.subplot(4,4,ti+1+8)
    plt.imshow(normalize(lfp_sparse_rect[:, :, ti, 0]).T, aspect='auto', cmap='bwr')

# %% Compute RMSE and R2
rmse = np.sqrt(np.mean(np.square(csd_pred_rect - csd_dense_rect)))
r2 = 1 - np.sum(np.square(csd_pred_rect - csd_dense_rect))/np.sum(np.square(csd_dense_rect))
# %%

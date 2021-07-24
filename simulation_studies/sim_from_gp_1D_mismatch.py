"""
Simulate some data from GP to test methods and evaluate errors.
In this case, the fitted GPCSD model does not match the generating model form.
Results are described in paper Supplement.

"""

# %% Imports
import matplotlib.pyplot as plt
import autograd.numpy as np
import scipy.interpolate
import scipy.stats

from gpcsd.gpcsd1d import GPCSD1D
from gpcsd.covariances import *
from gpcsd.predict_csd import predictcsd_trad_1d
from gpcsd.utility_functions import normalize

np.random.seed(42)

 # %%
# Setup
ntrials = 50

a = 0
b = 2300
nt = 60
t = np.linspace(0, nt, nt)[:, None]

nx = 24
x = np.linspace(a, b, nx)[:, None]
deltax = x[1] - x[0]
xshort = x[1:-1]
nz = 100
z = np.linspace(a, b, nz)[:, None]
deltaz = z[1] - z[0]

# Case 1: two components (SE with Matern noise), will fit with SE only and white noise
gpcsd_gen2 = GPCSD1D(np.zeros((nz, nt)), z, t, temporal_cov_list=[GPCSDTemporalCovSE(t), GPCSDTemporalCovMatern(t)])
gpcsd_gen2.R['value'] = 200
gpcsd_gen2.sig2n['value'] = 0.0001
gpcsd_gen2.spatial_cov.params['ell']['value'] = 200
gpcsd_gen2.temporal_cov_list[0].params['ell']['value'] = 20.0
gpcsd_gen2.temporal_cov_list[0].params['sigma2']['value'] = 1.5
gpcsd_gen2.temporal_cov_list[1].params['ell']['value'] = 2.0
gpcsd_gen2.temporal_cov_list[1].params['sigma2']['value'] = 0.2

# Case 2: three components (two SE with Matern noise), will fit with two-component model
gpcsd_gen3 = GPCSD1D(np.zeros((nz, nt)), z, t, temporal_cov_list=[GPCSDTemporalCovSE(t), GPCSDTemporalCovSE(t), GPCSDTemporalCovMatern(t)])
gpcsd_gen3.R['value'] = 200
gpcsd_gen3.sig2n['value'] = 0.0001
gpcsd_gen3.spatial_cov.params['ell']['value'] = 200
gpcsd_gen3.temporal_cov_list[0].params['ell']['value'] = 10.0
gpcsd_gen3.temporal_cov_list[0].params['sigma2']['value'] = 0.5
gpcsd_gen3.temporal_cov_list[1].params['ell']['value'] = 100.0
gpcsd_gen3.temporal_cov_list[1].params['sigma2']['value'] = 0.1
gpcsd_gen3.temporal_cov_list[2].params['ell']['value'] = 2.0
gpcsd_gen3.temporal_cov_list[2].params['sigma2']['value'] = 0.2

# %% Generate CSD and sample at interior electrode positions for comparing to tCSD
csd2 = gpcsd_gen2.sample_prior(ntrials)
csd2_interior_electrodes = np.zeros((nx-2, nt, ntrials))
csd3 = gpcsd_gen3.sample_prior(ntrials)
csd3_interior_electrodes = np.zeros((nx-2, nt, ntrials))
for trial in range(ntrials):
    csdinterp = scipy.interpolate.RectBivariateSpline(z, t, csd2[:, :, trial])
    csd2_interior_electrodes[:, :, trial] = csdinterp(xshort, t)
    csdinterp = scipy.interpolate.RectBivariateSpline(z, t, csd3[:, :, trial])
    csd3_interior_electrodes[:, :, trial] = csdinterp(xshort, t)

# %% Pass through forward model, add white noise
lfp2 = np.zeros((nx, nt, ntrials))
lfp3 = np.zeros((nx, nt, ntrials))
for trial in range(ntrials):
    lfp2[:, :, trial] = fwd_model_1d(csd2[:, :, trial], z, x, 100) 
    lfp3[:, :, trial] = fwd_model_1d(csd3[:, :, trial], z, x, 100) 
lfp2 = lfp2 + np.random.normal(0, np.sqrt(0.0001), size=(nx, nt, ntrials))
lfp3 = lfp3 + np.random.normal(0, np.sqrt(0.0001), size=(nx, nt, ntrials))
lfp2 = normalize(lfp2)
lfp3 = normalize(lfp3)

# %%
plt.figure(figsize=(8, 5))
plt.subplot(121)
plt.imshow(normalize(csd2[:, :, 0]), vmin=-1, vmax=1, cmap='bwr', aspect='auto')
plt.xlabel('Time')
plt.ylabel('Depth')
plt.title('Two-component CSD')
plt.colorbar()
plt.subplot(122)
plt.imshow(lfp2[:, :, 0], cmap='bwr', aspect='auto')
plt.xlabel('Time')
plt.title('Two-component LFP')
plt.colorbar()
plt.show()

# %%
plt.figure(figsize=(8, 5))
plt.subplot(121)
plt.imshow(normalize(csd3[:, :, 0]), vmin=-1, vmax=1, cmap='bwr', aspect='auto')
plt.xlabel('Time')
plt.ylabel('Depth')
plt.title('Three-component CSD')
plt.colorbar()
plt.subplot(122)
plt.imshow(lfp3[:, :, 0], cmap='bwr', aspect='auto')
plt.xlabel('Time')
plt.title('Three-component LFP')
plt.colorbar()
plt.show()

# %% trad CSD on test data
tcsd_pred2 = predictcsd_trad_1d(lfp2)[1:-1, :, :]
tcsd_pred3 = predictcsd_trad_1d(lfp3)[1:-1, :, :]

# %% Fit GPCSD: one component when two were used to generate data (misspecified noise model)
SE_cov = GPCSDTemporalCovSE(t, ell_prior=GPCSDInvGammaPrior(), sigma2_prior=GPCSDHalfNormalPrior(sd=5))
SE_cov.params['ell']['prior'].set_params(20, 100)
gpcsd_model2 = GPCSD1D(lfp2, x, t, temporal_cov_list=[SE_cov], sig2n_prior=[GPCSDHalfNormalPrior(0.5) for i in range(nx)])
gpcsd_model2.R['value'] = 100
gpcsd_model2.fit()
print(gpcsd_model2)
gpcsd_model2.predict(xshort, t)

# %% Fit GPCSD: two components when three were used to generate data
matern_cov = GPCSDTemporalCovMatern(t, ell_prior=GPCSDInvGammaPrior(), sigma2_prior=GPCSDHalfNormalPrior(sd=5))
matern_cov.params['ell']['prior'].set_params(1, 20)
SE_cov = GPCSDTemporalCovSE(t, ell_prior=GPCSDInvGammaPrior(), sigma2_prior=GPCSDHalfNormalPrior(sd=5))
SE_cov.params['ell']['prior'].set_params(20, 100)
gpcsd_model3 = GPCSD1D(lfp3, x, t, temporal_cov_list=[SE_cov, matern_cov])
gpcsd_model3.R['value'] = 100
gpcsd_model3.fit()
print(gpcsd_model3)
gpcsd_model3.predict(xshort, t)


# %% GPCSD fit with one instead of two components
plt.figure(figsize=(12, 6))
plt.subplot(141)
plt.imshow(normalize(csd2_interior_electrodes[:, :, 0]), vmin=-1, vmax=1, cmap='bwr', aspect='auto')
plt.title('CSD true')
plt.xlabel('Time (ms)')
plt.ylabel('Depth (1000 microns)')
plt.subplot(142)
plt.imshow(normalize(gpcsd_model2.csd_pred[:, :, 0]), vmin=-1, vmax=1, cmap='bwr', aspect='auto')
plt.title('GPCSD prediction')
plt.xlabel('Time (ms)')
plt.subplot(143)
plt.imshow(normalize(tcsd_pred2[:, :, 0]), vmin=-1, vmax=1, cmap='bwr', aspect='auto')
plt.title('Trad CSD prediction')
plt.xlabel('Time (ms)')
plt.subplot(144)
plt.imshow(normalize(lfp2[:, :, 0]), vmin=-1, vmax=1, cmap='bwr', aspect='auto')
plt.title('LFP')
plt.xlabel('Time (ms)')
plt.tight_layout()
plt.show()

# %% GPCSD fit with 2 instead of 3 components
plt.figure(figsize=(12, 6))
plt.subplot(141)
plt.imshow(normalize(csd3_interior_electrodes[:, :, 0]), vmin=-1, vmax=1, cmap='bwr', aspect='auto')
plt.title('CSD true')
plt.xlabel('Time (ms)')
plt.ylabel('Depth (1000 microns)')
plt.subplot(142)
plt.imshow(normalize(gpcsd_model3.csd_pred[:, :, 0]), vmin=-1, vmax=1, cmap='bwr', aspect='auto')
plt.title('GPCSD prediction')
plt.xlabel('Time (ms)')
plt.subplot(143)
plt.imshow(normalize(tcsd_pred3[:, :, 0]), vmin=-1, vmax=1, cmap='bwr', aspect='auto')
plt.title('Trad CSD prediction')
plt.xlabel('Time (ms)')
plt.subplot(144)
plt.imshow(normalize(lfp3[:, :, 0]), vmin=-1, vmax=1, cmap='bwr', aspect='auto')
plt.title('LFP')
plt.xlabel('Time (ms)')
plt.tight_layout()
plt.show()

# %%

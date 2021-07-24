"""
 Simulate some data with a simple template CSD (dipole-like with one spatial dimension), 
 compare GPCSD to traditional CSD and kCSD estimation with and without noise.
 Produces Figure 1 in the paper.

"""
# %%
import matplotlib.pyplot as plt
import autograd.numpy as np
import time

from gpcsd.gpcsd1d import GPCSD1D
from gpcsd import forward_models, predict_csd
from gpcsd.utility_functions import normalize

np.random.seed(1)

# %% Functions for CSD pattern and normalization
def csd_true_f(x, t):
    """
    Calculates a toy CSD mean function with two dipoles.
    :param x: desired spatial coordinates (between 0 and 24)
    :param t: desired time coordinates in milliseconds
    :return: (nx, nt) array of values
    """
    comp1 = np.exp(-(x-200)**2/(2*150**2))*np.exp(-(t.T-25)**2/(2*3**2)) 
    comp2 = -np.exp(-(x-800)**2/(2*150**2))*np.exp(-(t.T-25)**2/(2*4**2))
    comp3 = np.exp(-(x-1600)**2/(2*150**2))*np.exp(-(t.T-30)**2/(2*4**2)) 
    comp4 = -np.exp(-(x-2200)**2/(2*150**2))*np.exp(-(t.T-30)**2/(2*3**2))
    val = comp1 + comp2 + comp3 + comp4
    return val/np.max(np.abs(val))

# %% Setup 
a = 0          # Top edge of electrode probe (microns)
b = 2400       # Bottom edge of electrode probe (microns)
R_true = 150   # Radius of cylinder in forward model (microns)
deltaz = 1.0   # Spacing for predictions spatially (microns)
nt = 50        # Number of time points
nx = 24        # Number of observed spatial locations for LFP
snr = 30       # noise characteristics

t = np.linspace(0, 50, nt)[:, None]
x = np.linspace(a, b, nx)[:, None]
nz = int(np.rint(b-a)/deltaz) + 1
z = np.linspace(a, b, nz)[:, None]

# %% Create true CSD
csd_true = csd_true_f(z, t)
print('generated CSD of shape')
print(csd_true.shape)

# %% Create observed LFP
lfp = {}
lfp['noiseless'] = normalize(forward_models.fwd_model_1d(csd_true, z, x, R_true))
print('generated LFP of shape')
print(lfp['noiseless'].shape)

# Add white noise to LFP
sig2n_true = np.square(np.std(lfp['noiseless']) / snr)
lfp['white_noise'] = lfp['noiseless'] + np.random.normal(0, np.sqrt(sig2n_true), size=(nx, nt))

plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.imshow(csd_true, aspect='auto', cmap='bwr')
plt.title('True CSD')
plt.ylabel('Depth (microns)')
plt.xlabel('Time (ms)')
plt.subplot(132)
plt.imshow(lfp['noiseless'], aspect='auto', cmap='bwr')
plt.title('LFP')
plt.xlabel('Time (ms)')
plt.subplot(133)
plt.imshow(lfp['white_noise'], aspect='auto', cmap='bwr')
plt.title('LFP (noisy)')
plt.xlabel('Time (ms)')
plt.show()

# %% traditional CSD estimation
tcsd = {}
for k in lfp.keys():
    tcsd[k] = predict_csd.predictcsd_trad_1d(lfp[k][:, :, None]).squeeze()

# %% GPCSD model setup -- print prior information, parameter start values
gpcsd = {}
for k in lfp.keys():
    gpcsd[k] = GPCSD1D(lfp[k], x, t)
    print(gpcsd[k])  

# %% GPCSD fitting and predictions
for k in lfp.keys():
    print('\nStarting %s' % k)
    gpcsd[k].fit(n_restarts=10)
    gpcsd[k].predict(z, t)

# %% Print GPCSD fitting results
for k in lfp.keys():
    print(gpcsd[k])

# %% kCSD estimation
from kcsd import KCSD1D # https://github.com/Neuroinflab/kCSD-python/releases/tag/v2.0
start_t = time.process_time()
est_kcsd = {}
for k in lfp.keys():
    kcsd_tmp = KCSD1D(x, lfp[k], gdx=deltaz, h=R_true)
    kcsd_tmp.cross_validate(Rs=np.linspace(100, 800, 15), lambdas=np.logspace(1,-15,25,base=10.))
    est_kcsd[k] = kcsd_tmp.values()
end_t = time.process_time()
print('kCSD took %0.2f s (per dataset, with cross-validation)' % ((end_t - start_t)/len(est_kcsd)))

# %% Visualize results
vmlfp = np.amax(np.abs(lfp['noiseless']))
vmcsd = np.amax(np.abs(normalize(csd_true)))

plt.rcParams.update({'font.size': 12})

f = plt.figure(figsize=(16, 8))
ax = plt.subplot(251)
plt.imshow(lfp['noiseless'], aspect='auto', vmin=-vmlfp, vmax=vmlfp, cmap='bwr', extent=[t[0, 0], t[-1, 0], x[-1, 0], x[0, 0]])
plt.title('Noiseless LFP')
plt.ylabel('Depth')
axtmp = plt.subplot(252, sharey = ax)
plt.imshow(normalize(csd_true), aspect='auto', vmin=-vmcsd, vmax=vmcsd, cmap='bwr', extent=[t[0, 0], t[-1, 0], z[-1, 0], z[0, 0]])
plt.title('Ground truth CSD')
axtmp.set_yticklabels([])
plt.subplot(253, sharey = ax)
plt.imshow(normalize(tcsd['noiseless']), aspect='auto', vmin=-vmcsd, vmax=vmcsd, cmap='bwr', extent=[t[0, 0], t[-1, 0], x[-1, 0], x[0, 0]])
plt.title('tCSD')
plt.subplot(254, sharey = ax)
plt.imshow(normalize(gpcsd['noiseless'].csd_pred), aspect='auto', vmin=-vmcsd, vmax=vmcsd, cmap='bwr', extent=[t[0, 0], t[-1, 0], z[-1, 0], z[0, 0]])
plt.title('GPCSD')
plt.subplot(255, sharey = ax)
plt.imshow(normalize(est_kcsd['noiseless']), aspect='auto', vmin=-vmcsd, vmax=vmcsd, cmap='bwr', extent=[t[0, 0], t[-1, 0], z[-1, 0], z[0, 0]])
plt.title('kCSD')
plt.subplot(256, sharey = ax)
plt.imshow(lfp['white_noise'], aspect='auto', vmin=-vmlfp, vmax=vmlfp, cmap='bwr', extent=[t[0, 0], t[-1, 0], x[-1, 0], x[0, 0]])
plt.title('Noisy LFP')
plt.ylabel('Depth')
plt.xlabel('Time')
plt.subplot(257, sharey = ax)
plt.imshow(normalize(csd_true), aspect='auto', vmin=-vmcsd, vmax=vmcsd, cmap='bwr', extent=[t[0, 0], t[-1, 0], z[-1, 0], z[0, 0]])
plt.title('Ground truth CSD')
plt.xlabel('Time')
plt.subplot(258, sharey = ax)
plt.imshow(normalize(tcsd['white_noise']), aspect='auto', vmin=-vmcsd, vmax=vmcsd, cmap='bwr', extent=[t[0, 0], t[-1, 0], x[-1, 0], x[0, 0]])
plt.title('tCSD')
plt.xlabel('Time')
plt.subplot(259, sharey = ax)
plt.imshow(normalize(gpcsd['white_noise'].csd_pred), aspect='auto', vmin=-vmcsd, vmax=vmcsd, cmap='bwr', extent=[t[0, 0], t[-1, 0], z[-1, 0], z[0, 0]])
plt.title('GPCSD')
plt.xlabel('Time')
plt.subplot(2,5,10, sharey = ax)
im = plt.imshow(normalize(est_kcsd['white_noise']), aspect='auto', vmin=-vmcsd, vmax=vmcsd, cmap='bwr', extent=[t[0, 0], t[-1, 0], z[-1, 0], z[0, 0]])
plt.title('kCSD')
plt.xlabel('Time')
f.text(0.1, 0.85, 'A', fontsize=18)
f.text(0.1, 0.45, 'B', fontsize=18)
clb = f.colorbar(im, ax=f.axes)
clb.ax.set_title('a.u.')
plt.show()

# %%

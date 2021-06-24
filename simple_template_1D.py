"""
 Simulate some data with a simple template CSD (dipole-like one-dimensional), compare method performances.

"""
# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import invgamma, halfnorm

# TODO use package/module style imports
from forward_models import fwd_model_1d
from predict_csd import predictcsd_trad_1d
from utility_functions import inv_gamma_lim
import gpcsd_1d

def csd_true_mean_fn(x, t):
    """
    Calculates a toy CSD mean function with two dipoles.
    :param x: desired spatial coordinates (between 0 and 24)
    :param t: desired time coordinates in milliseconds
    :return: (nx, nt) array of values
    """
    val1 = np.exp(-(x-2)**2/(2*1.5**2))*np.exp(-(t.T-25)**2/(2*3**2)) - np.exp(-(x - 8) ** 2 / (2 * 1.5 ** 2)) * np.exp(-(t.T - 25) ** 2 / (2 * 4 ** 2))
    val2 = np.exp(-(x-16)**2/(2*1.5**2))*np.exp(-(t.T-30)**2/(2*4**2)) - np.exp(-(x - 22) ** 2 / (2 * 1.5 ** 2)) * np.exp(-(t.T - 30) ** 2 / (2 * 3 ** 2))
    val = val1 + val2
    return val

def normalize(x):
    return x/np.std(x)

np.random.seed(42)

# Setup
a = 0
b = 24
nt = 50
t = np.linspace(0, 50, nt)[:, None]
R_true = 2.
nx = 24
x = np.linspace(a, b, nx)[:, None]
nz = 100
z = np.linspace(a-2, b+2, nz)[:, None]
deltaz = z[1] - z[0]

# Create true mean
csd_true = csd_true_mean_fn(z, t)
print('generated CSD of shape')
print(csd_true.shape)
csd_true_coarse = csd_true_mean_fn(x, t)

# Pass through forward model
lfp_true = fwd_model_1d(csd_true, z, x, R_true)
print('generated LFP of shape')
print(lfp_true.shape)

# Add white noise to LFP
sig2n_true = 0.001
lfp_true_wn = lfp_true + np.random.normal(0, np.sqrt(sig2n_true), size=(nx, nt))

# TODO: try noise with temporal/spatial structure

### traditional CSD
tcsd_pred = predictcsd_trad_1d(lfp_true[:, :, None])
tcsd_pred = normalize(np.squeeze(tcsd_pred))
tcsd_pred_wn = predictcsd_trad_1d(lfp_true_wn[:, :, None])
tcsd_pred_wn = normalize(np.squeeze(tcsd_pred_wn))

### TODO KCSD using new kCSD package

# %% GPCSD 

# Set up priors and objective function

aR, bR = inv_gamma_lim(0.1, 10.)
aSE, bSE = inv_gamma_lim(1., 24.)
atSE, btSE = inv_gamma_lim(0.1, 60)
atM, btM = inv_gamma_lim(0.01, 1)

gpcsd_noiseless = gpcsd_1d.GPCSD_1D(lfp_true, x, t)
gpcsd_noiseless.fit([aR, bR, aSE, bSE, atSE, btSE, atM, btM], verbose=True)
gpcsd_noiseless.predict(z, t)

gpcsd_noisy = gpcsd_1d.GPCSD_1D(lfp_true_wn, x, t)
gpcsd_noisy.fit([aR, bR, aSE, bSE, atSE, btSE, atM, btM], verbose=True)
gpcsd_noisy.predict(z, t)

# %% Plot result TODO LFP in different color?
csd_true = normalize(csd_true)

vmlfp = np.amax(np.abs(lfp_true))
vmcsd = np.amax(np.abs(csd_true))
#vmgpcsd = np.amax(np.abs(gpcsd_pred))
#vmtcsd = np.amax(np.abs(tcsd_pred))
#vmkcsd = np.amax(np.abs(kcsd_pred))

f = plt.figure()
ax = plt.subplot(251)
plt.imshow(lfp_true, aspect='auto', vmin=-vmlfp, vmax=vmlfp, cmap='bwr', extent=[t[0, 0], t[-1, 0], x[-1, 0], x[0, 0]])
plt.title('Noiseless LFP')
plt.ylabel('Depth')
plt.subplot(252, sharey = ax)
plt.imshow(csd_true, aspect='auto', vmin=-vmcsd, vmax=vmcsd, cmap='bwr', extent=[t[0, 0], t[-1, 0], z[-1, 0], z[0, 0]])
plt.title('Ground truth CSD')
plt.subplot(253, sharey = ax)
plt.imshow(tcsd_pred, aspect='auto', vmin=-vmcsd, vmax=vmcsd, cmap='bwr', extent=[t[0, 0], t[-1, 0], x[-1, 0], x[0, 0]])
plt.title('tCSD')
plt.subplot(254, sharey = ax)
plt.imshow(gpcsd_noiseless.csd, aspect='auto', vmin=-vmcsd, vmax=vmcsd, cmap='bwr', extent=[t[0, 0], t[-1, 0], z[-1, 0], z[0, 0]])
plt.title('GPCSD')
plt.subplot(255, sharey = ax)
#plt.imshow(kcsd_pred, aspect='auto', vmin=-vmcsd, vmax=vmcsd, cmap='bwr', extent=[t[0, 0], t[-1, 0], z[-1, 0], z[0, 0]])
plt.title('kCSD')
plt.subplot(256, sharey = ax)
plt.imshow(lfp_true_wn, aspect='auto', vmin=-vmlfp, vmax=vmlfp, cmap='bwr', extent=[t[0, 0], t[-1, 0], x[-1, 0], x[0, 0]])
plt.title('Noisy LFP')
plt.ylabel('Depth')
plt.xlabel('Time')
plt.subplot(257, sharey = ax)
plt.imshow(csd_true, aspect='auto', vmin=-vmcsd, vmax=vmcsd, cmap='bwr', extent=[t[0, 0], t[-1, 0], z[-1, 0], z[0, 0]])
plt.title('Ground truth CSD')
plt.xlabel('Time')
plt.subplot(258, sharey = ax)
plt.imshow(tcsd_pred_wn, aspect='auto', vmin=-vmcsd, vmax=vmcsd, cmap='bwr', extent=[t[0, 0], t[-1, 0], x[-1, 0], x[0, 0]])
plt.title('tCSD')
plt.xlabel('Time')
plt.subplot(259, sharey = ax)
plt.imshow(gpcsd_noisy.csd, aspect='auto', vmin=-vmcsd, vmax=vmcsd, cmap='bwr', extent=[t[0, 0], t[-1, 0], z[-1, 0], z[0, 0]])
plt.title('GPCSD')
plt.xlabel('Time')
plt.subplot(2,5,10, sharey = ax)
#im = plt.imshow(kcsd_pred_wn, aspect='auto', vmin=-vmcsd, vmax=vmcsd, cmap='bwr', extent=[t[0, 0], t[-1, 0], z[-1, 0], z[0, 0]])
plt.title('kCSD')
plt.xlabel('Time')
#clb = f.colorbar(im, ax=f.axes)
#clb.ax.set_title('a.u.')
f.text(0.09, 0.9, 'A', fontsize=18)
f.text(0.09, 0.47, 'B', fontsize=18)
f.set_size_inches(14, 8)
plt.show()
#plt.savefig('%s/plots/sim_1D_simpletemplate_preds.png'%savepath)
#plt.close()


# %%

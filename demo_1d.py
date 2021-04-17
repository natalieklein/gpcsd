"""
Demo code for 1d csd functions.
"""

import matplotlib.pyplot as plt
import numpy as np

from utility_functions import plot_im
from forward_models import fwd_model_1d
from gp_cov import compKs_1d, compKt
from predict_csd import predictcsd_trad_1d, predictcsd_1d


# Set up ground truth GP and forward model parameters
R = 150       # cylinder radius, microns
ellSE = 300   # Spatial lengthscale
sig2tM = 10   # Matern temporal marginal variance
elltM = 10    # Matern temporal lengthscale
sig2tSE = 80  # SE temporal marginal variance
elltSE = 5    # SE temporal lengthscale
sig2n = 0.5   # White noise variance

# Set up spatial locations, temporal locations
x = np.linspace(0, 2300, 24)[:, None]  # LFP spatial locations, microns
z = np.linspace(0, 2300, 100)[:, None] # CSD spatial locations, microns
t = np.linspace(0, 100, 100)[:, None]  # time points, milliseconds

# Generate ground truth CSD from Gaussian process
Ks_csd = compKs_1d(z, ellSE)
Kt_res = compKt(t, t, sig2tM, elltM, sig2tSE, elltSE)
Kt = Kt_res[0] + Kt_res[1]
Lt = np.linalg.cholesky(Kt)
Ls = np.linalg.cholesky(Ks_csd + 1e-7*np.eye(z.shape[0]))
csd = np.dot(np.dot(Ls, np.random.normal(0, 1, (z.shape[0], t.shape[0]))), Lt.T)

csd = np.exp( -np.square((z-600)/300) -np.square((t.T-30)/5) ) \
- 1.5*np.exp( -np.square((z-1000)/300) -np.square((t.T-30)/5) ) \
+ 1.5*np.exp( -np.square((z-1500)/300) -np.square((t.T-35)/5) ) \
- np.exp( -np.square((z-1900)/300) -np.square((t.T-35)/5) )

# Pass through forward model
lfp = fwd_model_1d(csd, z, x, R)

# predict CSD using traditional CSD and GPCSD with true GP hyperparameters
tcsd_pred = predictcsd_trad_1d(lfp[:, :, None])
gpcsd_SE, gpcsd_M = predictcsd_1d(R, ellSE, sig2tM, elltM, sig2tSE, elltSE, sig2n, z, t, x, t, lfp[:, :, None],
                                  a=np.min(x)-100, b=np.max(x)+100)
gpcsd_pred = gpcsd_SE + gpcsd_M

# Plot CSD and LFP
plt.subplot(141)
plot_im(csd, t, z)
plt.xlabel('Time (ms)')
plt.ylabel('Depth (microns)')
plt.title('CSD')
plt.subplot(142)
plot_im(gpcsd_pred.squeeze(), t, z)
plt.xlabel('Time (ms)')
plt.ylabel('Depth (microns)')
plt.title('GPCSD')
plt.subplot(143)
plot_im(tcsd_pred.squeeze(), t, x)
plt.xlabel('Time (ms)')
plt.ylabel('Depth (microns)')
plt.title('Trad CSD')
plt.subplot(144)
plot_im(lfp, t, x)
plt.xlabel('Time (ms)')
plt.ylabel('Depth (microns)')
plt.title('LFP')
plt.show()
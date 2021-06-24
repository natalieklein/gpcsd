"""
Demo code for 2d csd functions.
"""

import matplotlib.pyplot as plt
import numpy as np

from utility_functions import plot_im, expand_grid
from forward_models import fwd_model_2d
from gp_cov import compKs_2d, compKt
from predict_csd import predictcsd_trad_2d, predictcsd_2d


# Set up ground truth GP and forward model parameters
R = 30                        # constant-CSD slab width, microns
eps = 10                      # fwd model singularity param
ellSE1 = 40                   # Spatial lengthscale 1 (depth)
ellSE2 = 100                  # Spatial lengthscale 2 (width)
sig2tM = 10                   # Matern temporal marginal variance
elltM = 1                     # Matern temporal lengthscale
sig2tSE = 20                  # SE temporal marginal variance
elltSE = 5                    # SE temporal lengthscale
sig2n = 0.5                   # White noise variance

# Set up spatial locations, temporal locations
# Uniform grid z values for simulated CSD
nz1 = 20
nz2 = 500
z1 = np.linspace(0, 60, nz1)[:,None]
z2 = np.linspace(0, 1500, nz2)[:,None]
z = expand_grid(z1, z2)
nz = z.shape[0]
# Grid for LFP coarser resolution
nx1 = 6
nx2 = 60
x1 = np.linspace(0, 60, nx1)[:,None]
x2 = np.linspace(0, 1500, nx2)[:,None]
x = expand_grid(x1, x2)
nx = x.shape[0]
nt = 10
t = np.linspace(0, 100, nt)[:, None]  # time points, milliseconds

# Generate ground truth CSD from Gaussian process
Ks_csd = compKs_2d(z, ellSE1, ellSE2)
Kt_res = compKt(t, t, sig2tM, elltM, sig2tSE, elltSE)
Kt = Kt_res[0] + Kt_res[1]
Lt = np.linalg.cholesky(Kt)
Ls = np.linalg.cholesky(Ks_csd + 1e-7*np.eye(z.shape[0]))
csd = np.dot(np.dot(Ls, np.random.normal(0, 1, (z.shape[0], t.shape[0]))), Lt.T)

# Pass through forward model
lfp = fwd_model_2d(csd.reshape((nz1, nz2, -1)), z1, z2, x, R, eps).reshape((nx1, nx2, nt))

# predict CSD using traditional CSD and GPCSD with true GP hyperparameters
tcsd_pred = predictcsd_trad_2d(lfp[:, :, :, None]).squeeze()
gpcsd_SE, gpcsd_M = predictcsd_2d(R, ellSE1, ellSE2, sig2tM, elltM, sig2tSE, elltSE, sig2n, z, t, x, t,
                                  lfp.reshape((nx, nt))[:, :, None], np.min(x1)-50, np.max(x1)+50, np.min(x2)-100,
                                  np.max(x2)+100, eps, 60, 150)
gpcsd_pred = gpcsd_SE + gpcsd_M
gpcsd_pred = gpcsd_pred.reshape((nz1, nz2, nt))
csd = csd.reshape((nz1, nz2, nt))

# Plot CSD and LFP over time
nt_plot = 6
counter = 1
f = plt.figure(figsize=(15, 12))
for ti in range(nt_plot):
    plt.subplot(4, nt_plot, counter)
    plot_im(csd[:, :, ti].T, z1, z2)
    plt.xlabel('Width (microns)')
    plt.ylabel('Depth (microns)')
    plt.title('CSD t=%0.2f'%t[ti])
    counter += 1
for ti in range(nt_plot):
    plt.subplot(4, nt_plot, counter)
    plot_im(gpcsd_pred[:, :, ti].T, z1, z2)
    plt.xlabel('Width (microns)')
    plt.ylabel('Depth (microns)')
    plt.title('GPCSD t=%0.2f'%t[ti])
    counter += 1
for ti in range(nt_plot):
    plt.subplot(4, nt_plot, counter)
    plot_im(tcsd_pred[:, :, ti].T, x1, x2)
    plt.xlabel('Width (microns)')
    plt.ylabel('Depth (microns)')
    plt.title('Trad CSD t=%0.2f'%t[ti])
    counter += 1
for ti in range(nt_plot):
    plt.subplot(4, nt_plot, counter)
    plot_im(lfp[:, :, ti].T, x1, x2)
    plt.xlabel('Width (microns)')
    plt.ylabel('Depth (microns)')
    plt.title('LFP t=%0.2f'%t[ti])
    counter += 1
plt.tight_layout()
plt.show()

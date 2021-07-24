import numpy as np

def predictcsd_trad_1d(lfp):
    """
    Does traditional CSD estimator with no smoothing (may want to smooth LFP first)
    :param lfp: (nx, nt, ntrial)
    :return: (nx, nt, ntrial) values of CSD
    """
    nx = lfp.shape[0]
    nt = lfp.shape[1]
    ntrials = lfp.shape[2]
    csd = np.zeros((nx, nt, ntrials))
    for x in range(1,nx-1):
        csd[x, :, :] = lfp[x+1, :, :] + lfp[x-1, :, :] - 2*lfp[x, :, :]
    return -csd


def predictcsd_trad_2d(lfp):
    """
    Does traditional CSD estimator with no smoothing, columnwise, assuming data already on
    grid (interpolate to grid if needed). May also want to pre-smooth data.
    :param lfp: (nx1, nx2, nt, ntrial)
    :return: (nx1, nx2, nt, ntrial) values of CSD
    """
    nx1, nx2, nt, ntrials = lfp.shape
    csd = np.nan*np.ones((nx1, nx2, nt, ntrials))
    for row in range(nx1):
        lfprow = lfp[row, :, :, :]
        for col in range(1, nx2-1):
            csd[row, col, :, :] = lfprow[col+1, :, :] + lfprow[col-1, :, :] - 2*lfprow[col, :, :]
    return -csd
"""
Priors for GPCSD parameters.

"""

import autograd.numpy as np
from scipy.stats import invgamma, halfnorm

class GPCSDPrior:
    def __init__(self):
        pass
    

class GPCSDInvGammaPrior(GPCSDPrior):
    def __init__(self, alpha=1, beta=1):
        GPCSDPrior.__init__(self)
        self.alpha = alpha
        self.beta = beta

    def __str__(self):
        return "InvGamma(%0.2f, %0.2f)" % (self.alpha, self.beta)

    def lpdf(self, x):
        if x <= 0:
            val = -1.0 * np.inf
        else:
            val = -1.0 * (self.alpha + 1) * np.log(x) - self.beta/x
        return val

    def set_params(self, l, u):
        self.alpha = 2 + 9 * np.square((l + u)/(u - l))
        self.beta = 0.5 * (self.alpha - 1) * (l + u)
    
    def sample(self):
        return invgamma.rvs(self.alpha, scale=self.beta)


class GPCSDHalfNormalPrior(GPCSDPrior):
    def __init__(self, sd=1):
        GPCSDPrior.__init__(self)
        self.sd = sd

    def __str__(self):
        return "HalfNormal(%0.2f)" % (self.sd)

    def lpdf(self, x):
        if x <= 0:
            val = -1.0 * np.inf
        else:
            val = -0.5 * np.square(x / self.sd)
        return val
    
    def sample(self):
        return halfnorm.rvs(scale=self.sd)
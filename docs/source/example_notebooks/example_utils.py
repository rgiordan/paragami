import autograd
import autograd.numpy as np
import autograd.scipy as sp

def regularizer(par_dict, lam):
    return lam * np.sum(par_dict['mu'] ** 2)

def get_normal_log_prob(x, sigma, mu):
    sigma_inv = np.linalg.inv(sigma)
    sigma_det_sign, sigma_log_det = np.linalg.slogdet(sigma)
    if sigma_det_sign <= 0:
        return np.full(float('inf'), x.shape[0])
    else:
        x_centered = x - np.expand_dims(mu, axis=0)
        return -0.5 * (
            np.einsum('ni,ij,nj->n', x_centered, sigma_inv, x_centered) + \
            sigma_log_det)

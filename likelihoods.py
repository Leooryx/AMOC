import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from constants import *


# TODO: bien revérifier les maths derrière les likelihoods
# TODO: est-ce que je devrai pas changer les paramètres 
# pour se fixer aux paramètres réellement estimés ?
# params=(mu_0, rho, gamma^2, A, tau_r) ???



# 1. Ornstein-Uhlenbeck likelihood on stationary data. 

def OU_negloglik(pars):
    data = statio_data
    
    alpha0 = max(pars[0], 0.001)
    mu0    = pars[1]
    sigma2 = max(0, pars[2]) 
    n      = len(data)
    
    Xupp   = data[1:]
    Xlow   = data[:-1]
    
    gamma2 = sigma2 / (2 * alpha0)
    rho   = np.exp(-alpha0 * delta_t)
    
    Xupp   = data[1:]
    Xlow   = data[:-1]
    
    
    m_part = Xupp - Xlow * rho - mu0 * (1 - rho) #observation minus the mean of transition distribution
    v_part = gamma2 * (1 - rho**2) # variance of transition distribution
    
    loglik = (- n * np.log(v_part) - np.sum(m_part**2 / v_part)) #* (bounds["a"][0] <= a <= bounds["a"][1] and bounds["tau"][0] <= tau <= bounds["tau"][1])
    return -loglik


# 2. Pseudo-likelihood (Strang splitting) on non-stationary data

# unlike the article's code, here we use the pseudo likelihood to estimate all params
def pseudo_negloglik(pars):
    
    data = nonstatio_data

    # these 3 params were pre-computed via plug-in (cf. article)
    alpha = pars[0]
    mu    = pars[1]
    sigma2 = pars[2]
    
    tau = pars[3]
    a = pars[4]

    m = mu - alpha / (2 * a)
    lambda0 = -(alpha**2) / (4 * a)

    n = len(data)

    
    
    Xupp    = data[1:]
    Xlow    = data[:-1]
    
    time_seq = delta_t * np.arange(1, n)
    lam_seq    = lambda0 * (1 - time_seq / tau)
    alpha_seq  = 2 * np.sqrt(-a * lam_seq)
    gamma2_seq = sigma2 / (2 * alpha_seq)
    rho_seq    = np.exp(-alpha_seq * delta_t)
    mu_seq     = m + np.sqrt(-lam_seq / a)
    
    fh_half_tmp    = a * delta_t * (Xlow - mu_seq) / 2
    fh_half_tmpupp = a * delta_t * (Xupp - mu_seq) / 2
    
    fh_half     = (mu_seq * fh_half_tmp + Xlow) / (fh_half_tmp + 1)
    fh_half_inv = (mu_seq * fh_half_tmpupp - Xupp) / (fh_half_tmpupp - 1)
    
    mu_h        = fh_half * rho_seq + mu_seq * (1 - rho_seq)
    m_part      = fh_half_inv - mu_h
    var_part    = gamma2_seq * (1 - rho_seq**2)
    
    det_Dfh_half_inv = 1 / (a * delta_t * (Xupp - mu_seq) / 2 - 1)**2
    
    pen=0.004
    pen_term = pen * n * (1/a - 1) if a < 1 else 0
    
    loglik = - np.sum(np.log(var_part)) - np.sum(m_part**2 / var_part) + \
              2 * np.sum(np.log(det_Dfh_half_inv)) - pen_term
              
    return -loglik


def joint(pars): 
    alpha, mu, sigma2, tau, a = pars
    
    # ---------------------------------------------------------
    # PHASE 1 : Stationnaire (Ornstein-Uhlenbeck) pour t < t0
    # ---------------------------------------------------------
    Xupp_s, Xlow_s = statio_data[1:], statio_data[:-1]
    n_s = len(statio_data)
    
    gamma2 = sigma2 / (2 * alpha)
    rho = np.exp(-alpha * delta_t)
    
    m_part_s = Xupp_s - Xlow_s * rho - mu * (1 - rho)
    v_part_s = gamma2 * (1 - rho**2)
    
    # Log-vraisemblance OU (à une constante près)
    ll_ou = - n_s * np.log(v_part_s) - np.sum(m_part_s**2 / v_part_s)
    
    # ---------------------------------------------------------
    # PHASE 2 : Non-Stationnaire (Strang Splitting) pour t >= t0
    # ---------------------------------------------------------
    # Relations déterministes exactes du modèle
    m = mu - alpha / (2 * a)
    lambda0 = -(alpha**2) / (4 * a)
    
    Xupp_ns, Xlow_ns = nonstatio_data[1:], nonstatio_data[:-1]
    n_ns = len(nonstatio_data)
    time_seq = delta_t * np.arange(1, n_ns)
    
    # Évolution de lambda(t)
    lam_seq = lambda0 * (1 - time_seq / tau)
    
    """# Sécurité numérique : si les paramètres proposés par l'optimiseur
    # font basculer lambda en positif avant la fin des données, on rejette.
    if np.any(-a * lam_seq <= 0) or np.any(-lam_seq / a <= 0):
        return 1e10"""
        
    alpha_seq  = 2 * np.sqrt(-a * lam_seq)
    gamma2_seq = sigma2 / (2 * alpha_seq)
    rho_seq    = np.exp(-alpha_seq * delta_t)
    mu_seq     = m + np.sqrt(-lam_seq / a)
    
    fh_half_tmp    = a * delta_t * (Xlow_ns - mu_seq) / 2
    fh_half_tmpupp = a * delta_t * (Xupp_ns - mu_seq) / 2
    
    fh_half     = (mu_seq * fh_half_tmp + Xlow_ns) / (fh_half_tmp + 1)
    fh_half_inv = (mu_seq * fh_half_tmpupp - Xupp_ns) / (fh_half_tmpupp - 1)
    
    mu_h     = fh_half * rho_seq + mu_seq * (1 - rho_seq)
    m_part_ns= fh_half_inv - mu_h
    var_part_ns = gamma2_seq * (1 - rho_seq**2)
    
    det_Dfh = 1 / (a * delta_t * (Xupp_ns - mu_seq) / 2 - 1)**2
    
    pen_term = 0.004 * n_ns * (1/a - 1) if a < 1 else 0
    
    ll_ns = - np.sum(np.log(var_part_ns)) - np.sum(m_part_ns**2 / var_part_ns) + \
             2 * np.sum(np.log(det_Dfh)) - pen_term
             
    # La log-vraisemblance totale est la somme des deux phases
    joint_ll = ll_ou + ll_ns
    return -joint_ll  



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from constants import *
from utils import *


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



  
def joint_validified(pars):
    global statio_data, nonstatio_data
    
    BASELINE_MAX_LOGLIK = 5370.183291304657 
    M = 5000
    
    # 1. Observed log-likelihood ratio
    # joint(pars) returns the negative log-likelihood
    neg_ll_obs = joint(pars)
    log_R_obs = BASELINE_MAX_LOGLIK - (-neg_ll_obs)
    
    count_leq = 0
    
    # Save the original global data to restore it after simulations
    orig_statio = statio_data.copy()
    orig_nonstatio = nonstatio_data.copy()
    n_s = len(orig_statio)
    n_ns = len(orig_nonstatio)
    
    # Format parameters for your X_traj simulator
    p_dict = {
        "alpha": pars[0], 
        "mu": pars[1], 
        "sigma2": pars[2], 
        "tau": pars[3], 
        "a": pars[4],
        "lambda0": -(pars[0]**2) / (4 * pars[4]),
        "m": pars[1] - pars[0] / (2 * pars[4]),
        "t_c": t0 + pars[3] 
    }
    
    try:
        for _ in range(M):
            # 2. Simulate dataset using your exact function
            # Ymax ensures it runs long enough to cover both data phases
            sim_res = X_traj(p_dict, orig_statio[0], Ymax=n_s + n_ns + 10)
            sim_X = sim_res["X"]
            
            # Safety pad in case deterministic tip broke the loop early
            if len(sim_X) < n_s + n_ns:
                sim_X = np.pad(sim_X, (0, (n_s + n_ns) - len(sim_X)), mode='edge')
            
            # Temporarily replace global data so joint() evaluates the simulation
            statio_data = sim_X[:n_s]
            nonstatio_data = sim_X[n_s - 1 : n_s - 1 + n_ns]
            
            # 3. Simulated log-likelihood ratio
            neg_ll_sim_theta = joint(pars)
            
            # Compute MLE on this specific simulated dataset
            res = minimize(joint, x0=pars, method="Nelder-Mead", bounds=b(params))
            neg_ll_sim_mle = res.fun
            
            # log_R_sim = L(theta) - L(MLE)
            log_R_sim = (-neg_ll_sim_theta) - (-neg_ll_sim_mle)
            
            # 4. Validification check
            if log_R_sim <= log_R_obs:
                count_leq += 1
                
    finally:
        # Guarantee the original data is restored for the rest of your script
        statio_data = orig_statio
        nonstatio_data = orig_nonstatio
        
    pi_w = max(count_leq / M, 1e-12)
    
    # Return negative log-possibility to interface perfectly with your minimizers
    return -np.log(pi_w)



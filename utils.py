
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import cumulative_trapezoid
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import os
from scipy.interpolate import interp1d

from constants import *
from likelihoods import *


np.random.seed(123)


def S_onestep(sigma, lambda_val, m, a, Xt):
    dWt = np.random.normal(0,1)
    Xt_plus_1 = Xt - a*(Xt - m)**2 * delta_t - lambda_val * delta_t + sigma * np.sqrt(delta_t) * dWt
    return Xt_plus_1


def X_traj(params, X0, Ymax=1000000):

    deterministic_tip = False
    first_passage = False
    rebound = False
    
    alpha = params["alpha"]
    mu    = params["mu"]
    sigma2 = params["sigma2"]
    sigma = sigma2**0.5
    tau   = params["tau"]
    a     = params["a"]
    lambda0 = params["lambda0"] #need to concatenate dictionaries of estimated parameters!
    m = params["m"]
    t_c = params["t_c"]

    duration_bef_ramping = t0 - 1870
    
    Y = 0
    X = X0
    Xtraj = [X]
    barrier = m - np.sqrt(-lambda0 / a)

    # stationary phase
    while X > barrier and Y < duration_bef_ramping / delta_t:
        X = S_onestep(sigma=sigma, lambda_val=lambda0, m=m, a=a, Xt=X)
        Xtraj.append(X)
        Y += 1
    
    # non stationary phase
    while Y < Ymax:
        time = delta_t * (Y - duration_bef_ramping / delta_t)
        current_lambda = lambda0 * (1 - time / tau)

        if current_lambda >= 0: # means deterministic tipping
            deterministic_tip = True
            break

        barrier = m - np.sqrt(-current_lambda / a)
        X = S_onestep(sigma=sigma, lambda_val = current_lambda, m=m, a=a, Xt=X)
        Xtraj.append(X)
        Y += 1

        if X <= barrier and X > -3: #if happens before current_lambda >= 0, it is a noise-induced tipping
            first_passage = True
            #break 
            # we continue even after we pass the border
        
        if first_passage and X > barrier:
            rebound = True
    
    Y = Y * delta_t + 1870
    
    if Y > t_c: # we count Y in unit of month, it can therefore be above t_c, which is in absolute time unit
        Y = t_c
    
    
    return {"FPT": Y, "X" : np.array(Xtraj), "deterministic_tip": deterministic_tip, "rebound": rebound} # First Passage Time



def b(params):
    """Convenience: return a list of (lo, hi) tuples for given keys, in order."""
    return [bounds[k] for k in params]

def m_formula(mu, alpha, a):
    return mu - alpha / (2*a)

def lambda0_formula(alpha, a):
    return - (alpha**2) / (4*a)

def lambda_t_formula(t, lambda0, tau):
    return lambda0 * (1 - (t - t0 >= 0) * (t-t0) / tau)


def MLE(function, params):    
    
    res = minimize(function, init_params, method='Nelder-Mead', bounds=b(params)) 

    max_loglik = -res.fun
    best_params = dict(zip(params, res.x))
        
    alpha = best_params["alpha"]
    mu    = best_params["mu"]
    sigma2 = best_params["sigma2"]
    tau   = best_params["tau"]
    a     = best_params["a"]

    other_params = {'m' : m_formula(mu, alpha, a),
                    'lambda0' : lambda0_formula(alpha, a),
                    't_c' : tau + t0}
    
    #best_params.update(other_params)

    return max_loglik, best_params, other_params



def marg(function, params, max_loglik, best_params):
    """Returns a possibility function for each parameter"""

    profiles = {}

    for target_idx, target_param in enumerate(params):

        base_grid = intervals[target_param]
        best_val = best_params[target_param]
        
        grid = np.unique(np.append(base_grid, best_val))
        profiles[target_param] = np.zeros(len(grid))
        other_params = [p for p in params if p != target_param]
        other_bounds = [bounds[p] for p in other_params]
        other_inits = [best_params[p] for p in other_params]

        for i, val in enumerate(tqdm(grid, desc=f"{target_param.capitalize()} profile")):
            
            def obj(pars_free):
                full_pars = list(pars_free)
                full_pars.insert(target_idx, val)  # Re-insert fixed value at original index
                return function(full_pars)

            res = minimize(obj, other_inits, method='Nelder-Mead', bounds=other_bounds)

            profiles[target_param][i] = np.exp(-res.fun - max_loglik)


    return profiles



def credibility_intervals(profiles, intervals, func_name, threshold=0.25, filename="estimation_metrics.txt"):
    
    results = {}
    
    for param in profiles.keys():
        prof = profiles[param]
        grid = intervals[param]
        
        # Find indices where the profile is above the threshold
        valid_indices = np.where(prof >= threshold)[0]
        
        if len(valid_indices) > 0:
            # Take the first and last valid grid points as the bounds
            lower_bound = grid[valid_indices[0]]
            upper_bound = grid[valid_indices[-1]]
        else:
            # Fallback if the profile never reaches the threshold
            lower_bound, upper_bound = np.nan, np.nan 
            
        results[param] = (lower_bound, upper_bound)
        
    # 2. Write the results to the text file
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a') as f:
        # Create headers if the file is being created for the first time
        if not file_exists:
            headers = ["Likelihood_Function"]
            for param in profiles.keys():
                headers.append(f"{param}_lower")
                headers.append(f"{param}_upper")
            f.write("\t".join(headers) + "\n")
            
        # Construct and write the data row
        row_data = [func_name]
        for param in profiles.keys():
            lb, ub = results[param]
            row_data.append(f"{lb:.4f}")
            row_data.append(f"{ub:.4f}")
            
        f.write("\t".join(row_data) + "\n")
        
    print(f"Metrics successfully saved to {filename} for {func_name}.")
    
    return results



def U(x, Lambda, A, m):
    return A*(x-m)**3/3 + Lambda * x

def U_2nd(x, A, m):
    return 2*A*(x-m)


# TODO: mettre best_params dedans 
def mean_tau_noise_EK(Lambda, sigma, A, m):
    """Eyring-Kramers law: average time before noise tipping"""
    x_minus = m - np.sqrt(np.abs(Lambda)/A)
    x_plus = m + np.sqrt(np.abs(Lambda)/A)
    potential_barrier = U(x_minus, Lambda, A, m) - U(x_plus, Lambda, A, m) 
    numerator = 2*np.pi * np.exp(2*potential_barrier / sigma**2)
    denominator = np.sqrt(U_2nd(x_plus, A, m) * np.abs(U_2nd(x_minus, A, m)))

    return numerator / denominator


def proba_tipping_threshold(t, params, work_density=False):
    alpha = params["alpha"]
    mu    = params["mu"]
    sigma2 = params["sigma2"]
    sigma = sigma2**0.5
    tau   = params["tau"]
    a     = params["a"]
    m = m_formula(mu, alpha, a)
    lambda0 = lambda0_formula(alpha, a)
    t_c = tau + t0


    max_relative_time = int(min(t - t0, tau) + 1)
    time_seq = from_t0[from_t0 <= t0 + max_relative_time]

    lambda_grid = [lambda_t_formula(s, lambda0, tau) for s in time_seq]
    tau_grid = np.array([mean_tau_noise_EK(lam, sigma, a, m) for lam in lambda_grid])
    hazard_grid = 1 / tau_grid
    cumhaz = cumulative_trapezoid(hazard_grid, time_seq, initial=0)
    survival = np.exp(-cumhaz)

    cdf = 1 - survival
    cdf[time_seq >= t_c] = 1.0
    cdf_base = interp1d(time_seq, cdf, bounds_error=False, fill_value=(0.0, 1.0))

    def cdf_interp(t):
        t = np.asarray(t)
        return np.where(t >= t_c, 1.0, cdf_base(t))
    
    output = {'cdf': cdf_interp}

    if work_density:
        density = hazard_grid * survival
        density /= np.trapezoid(density, time_seq)
        density_base = interp1d(time_seq,density,bounds_error=False,fill_value=(0.0, 0.0))

        def density_interp(t):
            t = np.asarray(t)
            return np.where(t >= t_c, 0.0, density_base(t))
        
        output["density"] = density_interp
    return output


    

"""def possTip(t, likelihood, global_max_likelihood):

    params = ["alpha", "mu", "sigma2", "tau", "a"]


    def objective(pars):
        alpha, mu, sigma2, tau, a = pars
        
        if alpha <= 0 or sigma2 <= 0 or tau <= 0 or a <= 0:
            return 1e10
            
        neg_ll = likelihood(pars)
        rel_likelihood = np.exp(-(neg_ll - global_max_likelihood))
        

        pars_dict = {"alpha": alpha,"mu": mu,"sigma2": sigma2,"tau": tau,"a": a}
            
        cdf_func = proba_tipping(t, pars_dict)['cdf']
        p_tipping = float(cdf_func(t))
        
        return -(rel_likelihood * p_tipping)


    res = minimize(objective, init_params, method='L-BFGS-B', bounds=b(params))
    
    if not(res.success):
        return "error during optimization"
    
    return -res.fun"""


# useful?
def density_tau_noise_easy(t, Lambda, sigma, a, m):
    mean = mean_tau_noise_EK(Lambda, sigma, a, m)
    result = (1 / mean) * np.exp(-t / mean)
    return result






def hazard_rate(t, theta):
    alpha, mu, sigma2, tau, a = theta
    m = mu - alpha / (2 * a)
    lambda0 = -(alpha**2) / (4 * a)
    lam = lambda0 * (1 - t / tau)
    
    r = np.full_like(lam, np.inf, dtype=float)
    mask = lam < 0

    if np.any(mask):
        r[mask] = 1.0 / mean_tau_noise_EK(lam[mask], sigma2, a, m)
    return r

def possibility_theta_5d(theta, T, sup_loglik_global):
    """
    Evaluates possibility dynamically calculating t_c(theta) = tau.
    theta = [alpha, mu, sigma, tau, a]
    """
    alpha, mu, sigma2, tau, a = theta

    # 1. Epistemic uncertainty (Prior/Likelihood)
    negloglik = joint(theta)
    poss = np.exp(negloglik - sup_loglik_global)

    # 2. Aleatoric uncertainty (CDF)
    # --> do we work in relative or absolute time scale??? Tau is defined relatively. 
    #t_c = tau + t0

    if T >= tau:
        # If the target time T is beyond the deterministic tipping point, 
        # tipping has happened almost surely.
        cdf = 1.0
    else:
        # Evaluate noise-induced probability before deterministic tipping
        t_grid = np.linspace(0, T, 500)
        r = hazard_rate(t_grid, theta)
        H = cumulative_trapezoid(r, t_grid, initial=0)
        cdf = 1.0 - np.exp(-H[-1])

    return poss * cdf

params = ["alpha", "mu", "sigma2", "tau", "a"]

def possTip(T, best_theta_5d, sup_log_lik):
    """
    Finds the supremum of the combined epistemic and aleatoric probability.
    """
    if T <= 0:
        return {"possibility": 0.0, "best_theta_5d": best_theta_5d}

    def objective(theta):
        # We minimize the negative possibility
        return -possibility_theta_5d(theta, T, sup_log_lik)

    res = minimize(
        objective,
        x0=best_theta_5d,
        method="L-BFGS-B",
        bounds=b(params),
    )

    return {
        "possibility": -res.fun,
        "best_theta_5d": res.x,
        "optimizer": res,
    }




def simulate_trajectory(params, X0, t_start=1870, T_max=2080, dt=0.01):
    """
    Simulates the stochastic process and identifies tipping behaviors,
    accounting for rebounds that lead to either noise or bifurcation tipping.
    """
    a = params["a"]
    m = params["m"]
    lambda0 = params["lambda0"]
    tau = params["tau"]
    sigma = np.sqrt(params["sigma2"])
    
    t_c = t0 + tau     
    max_steps = int((T_max - t_start) / dt)
    
    X = np.zeros(max_steps)
    X[0] = X0
    dW = np.random.normal(0, 1, max_steps)
    
    # State tracking
    FPT = False
    rebound = False
    is_below = False
    classification_done = False
    
    # Final outcome flags
    bifurcation_tip = False
    noise_tip_before_tc = False
    
    for i in range(max_steps - 1):
        t = t_start + i * dt
        
        # 1. Compute lambda(t)
        if t < t0:
            lam_t = lambda0
        else:
            lam_t = lambda0 * (1 - (t - t0) / tau)
            
        # 2. Euler-Maruyama Step
        drift = - (a * (X[i] - m)**2 + lam_t)
        X[i+1] = X[i] + drift * dt + sigma * np.sqrt(dt) * dW[i]
        
        # 3. Escape condition
        if X[i+1] <= -3:
            X = X[:i+2] 
            break
            
        # 4. Tipping & Rebound Analysis
        if t < t_c:
            barrier = m - np.sqrt(max(0, -lam_t / a)) 
            
            # Crosses below the barrier
            if not is_below and X[i+1] <= barrier:
                is_below = True
                if FPT is False:
                    FPT = t + dt # Record FPT only on the very first crossing
            
            # Crosses back above the barrier (Rebound)
            elif is_below and X[i+1] > barrier:
                is_below = False
                rebound = True
                
        else:
            # 5. We reached t_c. Classify the final outcome based on current state.
            if not classification_done:
                classification_done = True
                
                if is_below:
                    # It was below the barrier when t_c hit
                    noise_tip_before_tc = True
                else:
                    # It survived above the barrier until t_c
                    bifurcation_tip = True
                    if FPT is False:
                        FPT = t_c

    # If the loop broke early (hit -3 before reaching t_c)
    if not classification_done:
        noise_tip_before_tc = True

    return {
        "FPT": FPT,
        "noise_tip_before_tc": noise_tip_before_tc,
        "rebound": rebound,
        "bifurcation_tip": bifurcation_tip,
        "X": X
    }
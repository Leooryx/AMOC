import numpy as np
from fp_transition import transition_densities_theta, lna_gaussian_pdf, theta_to_fp_params
from constants import *

"""
# Example
theta = (1.4, 1.7, 0.09, 20.0, 1.0)   # (alpha, mu, sigma2, tau, a)
t0 = 5.0

t_obs = np.array([0., 1., 2., 3., 4.])
x_obs = np.array([1.70, 1.65, 1.55, 1.40, 1.10])

sol = transition_densities_theta(theta, x_obs, t_obs, t0)  # solves all transitions at once

sol.loglik(x_obs[1:])        # true-model log-likelihood
sol.escaped                  # mass lost to -inf per transition (defect)

# plug in your own linearized/Strang density here instead of lna_gaussian_pdf
p = theta_to_fp_params(theta, t0)
q = lna_gaussian_pdf(p, x_obs[:-1], t_obs[:-1], t_obs[1:])
D_B = sol.bhattacharyya(q)   # Bhattacharyya distance per transition
f = np.exp(-D_B.sum())       # your f(theta | theta_tilde)

print(p)
print(q)
print(D_B)
print(f)

"""

# Real code


"""
Marginal (profile) possibility curves for the FLATTENED posterior.

This mirrors, as closely as possible, the marginal-profile code you already
have:

    params = ["alpha", "mu", "sigma2", "tau", "a"]
    function = joint
    max_loglik_global, best_params, other_params = MLE(function, params)
    profiles = marg(function, params, max_loglik_global, best_params)
    plot_profiles(profiles, best_params, f"{function.__name__}_marg.png")

`MLE`, `marg`, `plot_profiles`, `b`, `bounds`, `intervals`, `init_params`
and `joint` are NOT touched -- they are imported as-is. The only new piece
is the objective `function` itself: instead of `joint` (the plain
approximate-model negative log-likelihood), we plug in `joint_flattened`,
which is built from the possibilistic machinery in the tex note:

    tilde_L(theta|x) = sup_tilde_theta  f(theta|tilde_theta) * L(tilde_theta|x)
    f(theta|tilde_theta) = exp( -sum_k D_B(p_theta(.|x_{k-1}), q_tilde(.|x_{k-1})) )

p_theta is the TRUE model's transition density (fp_transition.py). q_tilde is
the closed-form approximate density: the linearized-OU density q for
transitions before T_SWITCH, the Strang-splitting density s from T_SWITCH
onward -- both built from the theta reparametrization (alpha, mu, sigma2,
tau, a) already used everywhere else in your code.
`joint(tilde_theta)` (imported, unmodified) already IS -log L(tilde_theta|x)
in that same q/s approximate world, so it is reused directly as the L(.|x)
term of the sup.

To keep the nested optimization cheap, D_B is only summed over a small,
evenly-spaced BUDGET of transitions (10 in the stationary regime, 25 in the
non-stationary one) rather than every observed transition -- the true-model
FP solve is run ONCE per outer theta (that's the expensive part); the inner
sup over tilde_theta only re-evaluates the closed-form q/s density and
`joint`, which are cheap.

--------------------------------------------------------------------------
SPEED (added, everything above/below is otherwise untouched):

  1. Caching. `joint_flattened` is wrapped in an exact-match `lru_cache`.
     L-BFGS-B (used inside both MLE and marg) frequently re-evaluates the
     exact same point -- most often when a parameter sits at/near a bound
     and several internal trial steps clip to the identical value, or
     during Nelder-Mead's own vertex bookkeeping in flattened_loglik. Those
     repeats are now free instead of re-running an FP solve. The cache key
     is the RAW, unrounded theta -- no rounding is used on purpose: L-BFGS-B
     estimates gradients with a finite-difference step of about 1e-8, so
     rounding to anything coarser than that (e.g. 6 decimals) would make a
     perturbed point collapse onto the base point and silently zero out the
     gradient. Exact-match caching cannot do that; it only ever reuses a
     result for a call that was already, bit-for-bit, made before.

  2. Parallelization. `marg`'s grid loop evaluates `grid_points` values for
     each of the 5 parameters, and every one of those (target_param, grid
     value) profile points is an independent 4-D optimization -- nothing is
     shared between them. `marg_parallel` below is the same loop, just
     handed out to a process pool (`concurrent.futures.ProcessPoolExecutor`,
     standard library, no new dependency) instead of run one point at a
     time. Only the final `profiles = marg(...)` call is swapped for
     `profiles = marg_parallel(...)`; `marg` itself (in utils.py) is not
     touched, so this still works if you ever want to switch back.

  3. `if __name__ == "__main__":` guard. This is required, not optional,
     for (2) to work correctly -- on Windows (and therefore on the Surface)
     `ProcessPoolExecutor` re-imports this file in every worker process. Without
     the guard, each worker would re-run the whole MLE/marg/plot_profiles
     driver at the bottom on import, spawning its own worker pool recursively.
     With the guard, a worker only picks up the function/constant definitions
     it needs and skips the driver, exactly as intended.

  Both changes are additive: nothing about `flattened_loglik`, `q_density`,
  `s_density`, `make_approx_pdf`, or the maths is touched.
--------------------------------------------------------------------------
"""

import os
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

from constants import *
from likelihoods import *
from utils import *
from plot_functions import *

from fp_transition import theta_to_fp_params, transition_densities





# ============================================================================
# 1) theta = (alpha, mu, sigma2, tau, a)  ->  local OU rate / mean at time t
#    (same reparametrization as fp_transition.theta_to_fp_params /
#    m_formula / lambda0_formula: alpha, mu are alpha(lambda0), mu(lambda0).)
# ============================================================================

def _unpack(theta):
    alpha, mu, sigma2, tau, a = theta
    m = mu - alpha / (2.0 * a)
    lambda0 = -(alpha ** 2) / (4.0 * a)
    return m, lambda0, tau, a, float(sigma2)


def _lambda_t(t, lambda0, tau):
    t = np.asarray(t, dtype=float)
    return np.where(t < t0, lambda0, lambda0 * (1.0 - (t - t0) / tau))


def _alpha_mu_t(t, theta, floor=-1e-10):
    """Local mean-reversion rate alpha(lambda_t) and stable mean mu(lambda_t),
    from linearizing b(x,t) = -(a(x-m)^2+lambda_t) around its stable root."""
    m, lambda0, tau, a, _ = _unpack(theta)
    lam_t = np.minimum(_lambda_t(t, lambda0, tau), floor)
    alpha_t = 2.0 * np.sqrt(-a * lam_t)
    mu_t = m + np.sqrt(-lam_t / a)
    return alpha_t, mu_t


# ============================================================================
# 2) q: linearized (non-stationary OU) transition density
# ============================================================================

def q_density(theta, x_prev, t_prev, h, x_next):
    """q_theta(t)(x_{t+h} | x_t), vectorized (broadcasts x_prev,t_prev,h against x_next)."""
    _, _, _, _, sigma2 = _unpack(theta)
    alpha_t, mu_t = _alpha_mu_t(t_prev, theta)
    gamma2_t = sigma2 / (2.0 * alpha_t)
    rho_t = np.exp(-alpha_t * h)
    m_t = x_prev * rho_t + mu_t * (1.0 - rho_t)
    Omega_t = np.maximum(gamma2_t * (1.0 - rho_t ** 2), 1e-13)
    return np.exp(-0.5 * (x_next - m_t) ** 2 / Omega_t) / np.sqrt(2 * np.pi * Omega_t)


# ============================================================================
# 3) s: Strang-splitting transition density
# ============================================================================

def _phi(s, x, a, mu_t):
    return mu_t + (x - mu_t) / (1.0 + a * s * (x - mu_t))


def _phi_inv(s, y, a, mu_t):
    denom = 1.0 - a * s * (y - mu_t)
    denom = np.where(np.abs(denom) < 1e-10, np.sign(denom) * 1e-10 + 1e-10, denom)
    return mu_t + (y - mu_t) / denom


def _phi_inv_jac(s, y, a, mu_t):
    denom = 1.0 - a * s * (y - mu_t)
    denom = np.where(np.abs(denom) < 1e-10, np.sign(denom) * 1e-10 + 1e-10, denom)
    return 1.0 / denom ** 2


def s_density(theta, x_prev, t_prev, h, x_next):
    """s_theta(t)(x_{t+h} | x_t) via Strang splitting (half-step Riccati flow,
    full-step linear OU, half-step Riccati flow back), vectorized."""
    _, _, _, a, sigma2 = _unpack(theta)
    alpha_t, mu_t = _alpha_mu_t(t_prev, theta)
    gamma2_t = sigma2 / (2.0 * alpha_t)
    rho_t = np.exp(-alpha_t * h)
    Omega_t = np.maximum(gamma2_t * (1.0 - rho_t ** 2), 1e-13)

    hs = h / 2.0
    y1 = _phi(hs, x_prev, a, mu_t)                 # phi_{h/2}(x_t)
    m_y1 = y1 * rho_t + mu_t * (1.0 - rho_t)        # m_t(phi_{h/2}(x_t))
    y2 = _phi_inv(hs, x_next, a, mu_t)              # (phi_{h/2})^{-1}(x_{t+h})
    Z = y2 - m_y1
    jac = np.abs(_phi_inv_jac(hs, x_next, a, mu_t))
    return np.exp(-0.5 * Z ** 2 / Omega_t) / np.sqrt(2 * np.pi * Omega_t) * jac


# ============================================================================
# 4) q/s switch, wrapped as the `q_pdf` callable TransitionSolution.bhattacharyya
#    expects: takes the (n,N) window array and returns density row by row.
# ============================================================================

def make_approx_pdf(theta, x_prev, t_prev, t_next):
    x_prev = np.asarray(x_prev, dtype=float)[:, None]
    t_prev_c = np.asarray(t_prev, dtype=float)[:, None]
    h = (np.asarray(t_next, dtype=float) - np.asarray(t_prev, dtype=float))[:, None]

    def pdf(x):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            q = q_density(theta, x_prev, t_prev_c, h, x)
            s = s_density(theta, x_prev, t_prev_c, h, x)
        out = np.where(t_prev_c < T_SWITCH, q, s)
        return np.nan_to_num(np.maximum(out, 0.0), nan=0.0, posinf=0.0)

    return pdf



# ============================================================================
# 5) flattened log-likelihood:  sup_tilde_theta [ -D_B_sum(theta,tilde) + (-joint(tilde)) ]
# ============================================================================

def flattened_loglik(theta, params, gamma=gamma):
    p = theta_to_fp_params(theta, t0)
    sol = transition_densities(p, X_PREV, T_PREV, T_NEXT, N=FP_N, n_steps=FP_STEPS)

    def inner_objective(tilde_theta):
        q_pdf = make_approx_pdf(tilde_theta, X_PREV, T_PREV, T_NEXT)
        D_B_sum = sol.bhattacharyya(q_pdf).sum()
        #print(D_B_sum)
        return gamma * D_B_sum / (N_STAT + N_NONSTAT +1)   + joint(list(tilde_theta))  #joint is the joint negloglik
        # plus for joint bc want to minimize

    res = minimize(inner_objective, x0=list(theta), method="Nelder-Mead",
                    bounds=b(params), options={"xatol": 1e-3, "fatol": 1e-3})
    
    return -res.fun


def _joint_flattened_uncached(theta):
    return -flattened_loglik(theta, params)


@lru_cache(maxsize=8192)
def _joint_flattened_cached(theta_key):
    # theta_key is a plain tuple of python floats -> hashable, picklable, and
    # re-importable in a worker process (see marg_parallel below).
    return _joint_flattened_uncached(theta_key)


def joint_flattened(theta):
    """Drop-in replacement for `joint`: NEGATIVE flattened log-likelihood,
    i.e. what MLE/marg expect to minimize.

    Wrapped in an exact-match cache (see module docstring, point 1): the key
    is the raw, unrounded theta, so this can only ever save a re-computation
    for a call that is bit-for-bit identical to an earlier one -- it never
    changes what gets computed, only how often.
    """
    theta_key = tuple(float(v) for v in theta)
    return _joint_flattened_cached(theta_key)



# ============================================================================
# 6) joint_true: negative log-likelihood built ENTIRELY from the true model's
# transition solver, over EVERY observed transition (X_OBS/T_OBS as given --
# not the small IDX subsample used for D_B in joint_flattened). No flattening,
# no sup over tilde_theta:
#     joint_true(theta) = -sum_k log p_theta(x_k | x_{k-1})
# MLE / marg / marg_parallel already turn -res.fun - max_loglik_global into
# exp(...), which IS the possibility-style relative likelihood
#     L(theta|x) = p(x|theta) / sup_psi p(x|psi)
# from the tex note -- joint_true itself doesn't need to normalize anything.
# ============================================================================

X_PREV_FULL, T_PREV_FULL, T_NEXT_FULL = X_OBS[:-1], T_OBS[:-1], T_OBS[1:]


def _true_loglik(theta):
    p = theta_to_fp_params(theta, t0)
    sol = transition_densities(p, X_PREV_FULL, T_PREV_FULL, T_NEXT_FULL, N=FP_N, n_steps=FP_STEPS)
    dens = sol.pdf(X_OBS[1:])
    return float(np.sum(np.log(np.maximum(dens, 1e-300))))


def _joint_true_uncached(theta):
    return -_true_loglik(theta)


@lru_cache(maxsize=8192)
def _joint_true_cached(theta_key):
    return _joint_true_uncached(theta_key)


def joint_true(theta):
    """Drop-in replacement for `joint`/`joint_flattened`: negative log-
    likelihood of the TRUE model alone, no approximate q/s, no flattening f,
    using every observed transition. Same exact-match-cache rationale as
    joint_flattened (see its docstring)."""
    theta_key = tuple(float(v) for v in theta)
    return _joint_true_cached(theta_key)


# ============================================================================
# 7) parallel drop-in for `marg`'s grid loop (see module docstring, point 2)
# ============================================================================

def _profile_point(args):
    """One (target_param, grid value) profile point -- independent of every
    other one, so safe to run in its own worker process."""
    function, target_idx, val, other_bounds, other_inits = args

    def obj(pars_free):
        full_pars = list(pars_free)
        full_pars.insert(target_idx, val)
        return function(full_pars)

    res = minimize(obj, other_inits, method="Nelder-Mead", bounds=other_bounds)
    return res.fun


def marg_parallel(function, params, max_loglik, best_params, max_workers=None):
    """Same computation, and same result, as `marg` (utils.py) -- only the
    `grid_points` x `len(params)` independent profile evaluations are farmed
    out to a process pool instead of run one after another. `marg` itself is
    left untouched."""
    if max_workers is None:
        # leave a core free, and don't ask for more than a handful -- fine
        # for a laptop-class machine (e.g. a Surface), where more workers
        # than physical cores mostly just adds contention/heat.
        max_workers = min(4, max(1, (os.cpu_count() or 2) - 1))

    tasks, task_keys = [], []
    for target_idx, target_param in enumerate(params):
        other_params = [p for p in params if p != target_param]
        other_bounds = [bounds[p] for p in other_params]
        other_inits = [best_params[p] for p in other_params]
        for i, val in enumerate(intervals[target_param]):
            tasks.append((function, target_idx, val, other_bounds, other_inits))
            task_keys.append((target_param, i))

    profiles = {p: np.zeros(grid_points) for p in params}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        results = tqdm(ex.map(_profile_point, tasks), total=len(tasks), desc="Profiles (parallel)")
        for (target_param, i), res_fun in zip(task_keys, results):
            profiles[target_param][i] = np.exp(-res_fun - max_loglik)

    return profiles




# ============================================================================
# 7) same marginal-profile pipeline as before, just with `function` swapped
#    and `marg` -> `marg_parallel`. Guarded by __main__: required for
#    ProcessPoolExecutor to work correctly (see module docstring, point 3).
# ============================================================================

if __name__ == "__main__":

    params = ["alpha", "mu", "sigma2", "tau", "a"]

    function = joint_true #joint_flattened
    print("let's go")
    
    name = "joint_true"
    import json

    MLE_FILE = f"{name}_mle.json"

    if os.path.exists(MLE_FILE):
        with open(MLE_FILE, "r") as f:
            saved = json.load(f)
        max_loglik_global = saved["max_loglik_global"]
        best_params = saved["best_params"]
        other_params = saved["other_params"]
        print(f"Loaded MLE result from {MLE_FILE}")
    else:
        max_loglik_global, best_params, other_params = MLE(function, params)
        with open(MLE_FILE, "w") as f:
            json.dump({
                "max_loglik_global": max_loglik_global,
                "best_params": best_params,
                "other_params": other_params,
            }, f, indent=2)
        print(f"Saved MLE result to {MLE_FILE}")

    print(best_params)
    print(other_params)

    profiles = marg_parallel(function, params, max_loglik_global, best_params)

    plot_profiles(profiles, best_params, f"{name}_marg.png")

    print(max_loglik_global)
    print(best_params)

import numpy as np
from fp_transition import transition_densities_theta, lna_gaussian_pdf, theta_to_fp_params

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
"""

import numpy as np
from scipy.optimize import minimize

from constants import *
from likelihoods import *
from utils import *
from plot_functions import *

from fp_transition import theta_to_fp_params, transition_densities


# ============================================================================
# 0) Observed data: plug in your actual arrays here.
#    x_obs[k] observed at t_obs[k], k = 0..n  (same series `joint` is fit on).
# ============================================================================

X_OBS = full_data   
T_OBS = full_time_seq    

assert X_OBS is not None and T_OBS is not None, (
    "Set X_OBS / T_OBS to the observed trajectory (values, times) that `joint` "
    "is already fit on, at the top of marginals_flattened.py."
)

T_SWITCH = 1924.0   # density switches from linearized (q) to Strang (s) here

# transition budget for D_B (kept small on purpose, see module docstring)
N_STAT, N_NONSTAT = 10, 25

# FP-solver resolution for the true-model transitions entering D_B
FP_N, FP_STEPS = 100, 50


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
# 5) small, fixed budget of transitions used for D_B
# ============================================================================

def _pick(idx, k):
    if len(idx) <= k:
        return idx
    pos = np.unique(np.round(np.linspace(0, len(idx) - 1, k)).astype(int))
    return idx[pos]


def select_transitions(t_obs, n_stat=N_STAT, n_nonstat=N_NONSTAT):
    t_prev_all = np.asarray(t_obs[:-1], dtype=float)
    stat_idx = np.where(t_prev_all < t0)[0]
    nonstat_idx = np.where(t_prev_all >= t0)[0]
    return np.sort(np.concatenate([_pick(stat_idx, n_stat), _pick(nonstat_idx, n_nonstat)]))


_IDX = select_transitions(T_OBS)
_X_PREV, _T_PREV, _T_NEXT = X_OBS[_IDX], T_OBS[_IDX], T_OBS[_IDX + 1]


# ============================================================================
# 6) flattened log-likelihood:  sup_tilde_theta [ -D_B_sum(theta,tilde) + (-joint(tilde)) ]
# ============================================================================

def flattened_loglik(theta, params):
    p = theta_to_fp_params(theta, t0)
    sol = transition_densities(p, _X_PREV, _T_PREV, _T_NEXT, N=FP_N, n_steps=FP_STEPS)

    def inner_objective(tilde_theta):
        q_pdf = make_approx_pdf(tilde_theta, _X_PREV, _T_PREV, _T_NEXT)
        D_B_sum = sol.bhattacharyya(q_pdf).sum()
        return D_B_sum + joint(list(tilde_theta))   # minimize D_B_sum - (-joint(tilde))

    res = minimize(inner_objective, x0=list(theta), method="Nelder-Mead",
                    bounds=b(params), options={"xatol": 1e-3, "fatol": 1e-3})
    return -res.fun


def joint_flattened(theta):
    """Drop-in replacement for `joint`: NEGATIVE flattened log-likelihood,
    i.e. what MLE/marg expect to minimize."""
    return -flattened_loglik(theta, params)


# ============================================================================
# 7) same marginal-profile pipeline as before, just with `function` swapped
# ============================================================================

params = ["alpha", "mu", "sigma2", "tau", "a"]

function = joint_flattened
max_loglik_global, best_params, other_params = MLE(function, params)

profiles = marg(function, params, max_loglik_global, best_params)

plot_profiles(profiles, best_params, f"{function.__name__}_marg.png")


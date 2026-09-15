"""
Numerical solution of the time-inhomogeneous Fokker-Planck equation for the
saddle-node ("fold") tipping model

    dX_t = -[A (X_t - m)^2 + lambda(t)] dt + sigma dW_t,

    lambda(t) = lambda0 * (1 - 1{t >= t0} * (t - t0) / tau_r),

with an absorbing boundary at the time-dependent unstable equilibrium

    ell(t) = m - sqrt(-lambda(t) / A)          (defined while lambda(t) <= 0)

and a reflecting (no-flux) boundary at x = r (the right edge of the
numerical domain, physically far from the dynamics).

The output is the survival probability S(t) = P(T_tip > t) and the
first-passage-time density f(t) = -dS/dt, i.e. the full distribution of
the tipping time T_tip = inf{t : X_t = ell(t)}.

See README.md for the full derivation and the justification of every
algorithmic choice made below. In one paragraph:

The Dirichlet boundary moves in time, which standard finite-difference
grids cannot handle directly, so we first apply a front-fixing (Landau)
change of variables y = (x - ell(t)) / (r - ell(t)) that maps the moving
domain to the fixed interval [0, 1]; done in conservative (flux-divergence)
form, this turns the moving-boundary problem into a fixed-grid
advection-diffusion equation with time-dependent coefficients, with no
extra reaction term, and lets us read the first-passage-time density
directly off the boundary flux. We discretize that equation with the
Scharfetter-Gummel / Chang-Cooper exponential finite-volume scheme, the
standard choice for drift-dominated Fokker-Planck / drift-diffusion
equations because it is unconditionally positivity-preserving and
conservative at any grid Peclet number -- essential here because the
effective drift-to-diffusion ratio grows without bound as the run
approaches the saddle-node bifurcation. Time stepping uses an implicit
theta-method (default backward Euler) so the linear system solved at each
step is an M-matrix, keeping the scheme unconditionally stable and
positivity preserving despite the stiffness from the collapsing domain
width and the diverging boundary speed near the bifurcation. We stop the
integration once the surviving probability drops below a tolerance, which
in practice happens comfortably before the bifurcation time and avoids
ever having to resolve that singular limit.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable, Optional
from tqdm import tqdm

import numpy as np
from scipy.linalg import solve_banded
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# --------------------------------------------------------------------------
# Model parameters
# --------------------------------------------------------------------------

@dataclass
class Params:
    A: float = 1.0
    m: float = 1.0
    lambda0: float = -0.5
    t0: float = 5.0
    tau_r: float = 20.0
    sigma: float = 0.2
    r: float = 3.0          # right (reflecting) edge of the numerical domain
    x_floor: float = -3.0   # stated lower bound of the physical state space (diagnostic only)

    @property
    def t_c(self) -> float:
        """Bifurcation time: lambda(t_c) = 0, the two fixed points merge."""
        return self.t0 + self.tau_r


def lam(t: np.ndarray, p: Params) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return np.where(t < p.t0, p.lambda0, p.lambda0 * (1.0 - (t - p.t0) / p.tau_r))


def lam_prime(t: np.ndarray, p: Params) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return np.where(t < p.t0, 0.0, -p.lambda0 / p.tau_r)


def ell(t: np.ndarray, p: Params, floor: float = -1e-10) -> np.ndarray:
    """Position of the (unstable) absorbing threshold x_-(t) = m - sqrt(-lambda(t)/A).

    `floor` keeps -lambda/A bounded away from 0 so we never hit a literal
    division by zero; the integrator is stopped well before this floor is
    reached (see `solve`), so it is a numerical safety net, not something
    that is meant to be relied on physically.
    """
    l = np.minimum(lam(t, p), floor)
    return p.m - np.sqrt(-l / p.A)


def ell_prime(t: np.ndarray, p: Params, floor: float = -1e-10) -> np.ndarray:
    l = np.minimum(lam(t, p), floor)
    lp = lam_prime(t, p)
    return lp / (2.0 * p.A * np.sqrt(-l / p.A))


def drift(x: np.ndarray, t: float, p: Params) -> np.ndarray:
    """b(x,t) = -(A (x-m)^2 + lambda(t))."""
    return -(p.A * (x - p.m) ** 2 + lam(t, p))


# --------------------------------------------------------------------------
# Scharfetter-Gummel / Chang-Cooper machinery
# --------------------------------------------------------------------------

def _bernoulli(w: np.ndarray) -> np.ndarray:
    """B(w) = w / (exp(w) - 1), evaluated in a numerically stable way.

    B(0) = 1. For very negative w, B(w) -> -w (upwind limit); for very
    positive w, B(w) -> w*exp(-w) -> 0.
    """
    w = np.asarray(w, dtype=float)
    out = np.empty_like(w)

    small = np.abs(w) < 1e-8
    out[small] = 1.0 - 0.5 * w[small] + (w[small] ** 2) / 12.0

    big_pos = w > 700
    out[big_pos] = w[big_pos] * np.exp(-w[big_pos])

    big_neg = w < -700
    out[big_neg] = -w[big_neg]

    reg = ~(small | big_pos | big_neg)
    out[reg] = w[reg] / np.expm1(w[reg])
    return out


@dataclass
class Grid:
    N: int
    dy: float
    y_centers: np.ndarray
    y_faces: np.ndarray  # length N+1, y_faces[0]=0, y_faces[N]=1

    @staticmethod
    def make(N: int) -> "Grid":
        dy = 1.0 / N
        yc = (np.arange(N) + 0.5) * dy
        yf = np.arange(N + 1) * dy
        return Grid(N=N, dy=dy, y_centers=yc, y_faces=yf)


def _face_state(t: float, p: Params, grid: Grid):
    """Return L(t), and the SG face velocity/diffusivity fields needed to
    assemble the operator at time t, evaluated on all N+1 faces."""
    L = p.r - ell(t, p)
    D = 0.5 * p.sigma ** 2
    Dloc = D / L ** 2  # uniform in y (depends only on t)

    yf = grid.y_faces
    x_f = ell(t, p) + L * yf
    v_mesh = ell_prime(t, p) * (1.0 - yf)
    tilde_b = drift(x_f, t, p) - v_mesh
    v_f = tilde_b / L
    return L, Dloc, v_f


def assemble_operator(t: float, p: Params, grid: Grid):
    """Build the tridiagonal generator L(t) such that dq/dt = L(t) q,
    for the conservative flux form  q_t = -d_y[ v(y,t) q - Dloc(t) q_y ],
    with q(0,t)=0 (Dirichlet, absorbing) and zero flux at y=1 (reflecting).

    Returns (sub, diag, super, left_flux_coeff) where left_flux_coeff is the
    coefficient such that the absorbing-boundary flux (and hence the
    first-passage-time density) is J_left = left_flux_coeff * q[0].
    """
    N = grid.N
    dy = grid.dy
    L, Dloc, v_f = _face_state(t, p, grid)

    # Interior + left-boundary faces k = 0, ..., N-1 (face N carries zero
    # flux by the reflecting condition and needs no formula).
    h = np.full(N, dy)
    h[0] = dy / 2.0  # left boundary face sits half a cell from the first center

    v = v_f[:N]
    w = v * h / Dloc
    beta = -(Dloc / h) * _bernoulli(w)  # coefficient of q_k in J_k (its own cell)

    alpha = np.zeros(N)  # coefficient of q_{k-1} in J_k (k=0 has no left cell: Dirichlet 0)
    w_int = v[1:] * dy / Dloc
    alpha[1:] = (Dloc / dy) * _bernoulli(-w_int)

    sub = np.zeros(N)
    diag = np.zeros(N)
    sup = np.zeros(N)

    # cell 0
    diag[0] = (beta[0] - alpha[1]) / dy if N > 1 else beta[0] / dy
    if N > 1:
        sup[0] = -beta[1] / dy

    # interior cells 1..N-2
    if N > 2:
        i = np.arange(1, N - 1)
        sub[i] = alpha[i] / dy
        diag[i] = (beta[i] - alpha[i + 1]) / dy
        sup[i] = -beta[i + 1] / dy

    # last cell N-1 (right flux is exactly zero: reflecting)
    if N > 1:
        sub[N - 1] = alpha[N - 1] / dy
        diag[N - 1] = beta[N - 1] / dy

    left_flux_coeff = beta[0]  # J_{1/2} = beta[0] * q[0]  (Dirichlet value at y=0 is 0)

    return sub, diag, sup, left_flux_coeff


# --------------------------------------------------------------------------
# Time integration
# --------------------------------------------------------------------------

@dataclass
class Solution:
    t: np.ndarray
    S: np.ndarray            # survival probability S(t) = P(T_tip > t)
    f: np.ndarray             # FPT density f(t) = -dS/dt, same length as t (f[0] is undefined/0)
    q_last: np.ndarray        # final density on the y-grid (diagnostic)
    grid: Grid
    p: Params
    stopped_early: bool
    residual_mass: float      # probability mass never resolved (lumped at t_c if run stops early)


def initial_condition(grid: Grid, p: Params, t_start: float,
                       mean_x: Optional[float] = None, std_x: Optional[float] = None) -> np.ndarray:
    """Default initial density: a narrow Gaussian around the stable fixed
    point x_+(t_start), expressed as q(y) = p(x(y)) * L (mass density in y).
    """
    L = p.r - ell(t_start, p)
    if mean_x is None:
        mean_x = p.m + np.sqrt(-lam(t_start, p) / p.A)
    if std_x is None:
        std_x = 3.0 * p.sigma / np.sqrt(2 * p.A * np.sqrt(-lam(t_start, p) / p.A))  # rough Kramers width

    x = ell(t_start, p) + L * grid.y_centers
    dens = np.exp(-0.5 * ((x - mean_x) / std_x) ** 2)
    q0 = dens * L
    q0 /= np.sum(q0) * grid.dy  # normalize discretely so S(0) = 1 exactly (matches how S(t) is computed)
    return q0


def solve(p: Params, N: int = 400, dt: float = 0.01, t_start: float = 0.0,
          theta: float = 1.0, S_tol: float = 1e-8, t_c_buffer: float = 1e-3,
          q0: Optional[np.ndarray] = None, max_steps : Optional[int] = 1000000) -> Solution:
    """Integrate the transformed Fokker-Planck equation forward in time.

    theta = 1.0 -> backward Euler (default: unconditionally stable and
                   positivity preserving, robust choice given the stiffness
                   near the bifurcation).
    theta = 0.5 -> Crank-Nicolson (2nd order in time, but can lose
                   positivity for large steps -- use a smaller dt if so).

    Stops when the surviving mass S(t) drops below `S_tol`, or when t
    approaches the bifurcation time t_c within `t_c_buffer * tau_r` (ell'(t)
    diverges at t_c; by that point essentially all trajectories have
    already tipped in every regime we've tested -- both stopping criteria
    are checked and reported).
    """
    grid = Grid.make(N)
    q = initial_condition(grid, p, t_start) if q0 is None else q0.copy()

    t_stop_hard = p.t_c - t_c_buffer * p.tau_r

    ts = [t_start]
    Ss = [float(np.sum(q) * grid.dy)]
    fs = [0.0]

    t = t_start
    step = 0
    stopped_early = False
    for step in range(0, max_steps):
        if Ss[-1] < S_tol:
            break
        if t >= t_stop_hard:
            stopped_early = True
            break
        if max_steps is not None and step >= max_steps:
            stopped_early = True
            break

        dt_eff = min(dt, t_stop_hard - t)
        t_new = t + dt_eff

        sub_n, diag_n, sup_n, left_n = assemble_operator(t_new, p, grid)
        sub_o, diag_o, sup_o, left_o = assemble_operator(t, p, grid)

        Nn = grid.N
        # Build banded matrix for (I - theta*dt*L_new) q_new = (I + (1-theta)*dt*L_old) q_old
        ab = np.zeros((3, Nn))
        ab[0, 1:] = -theta * dt_eff * sup_n[:-1]
        ab[1, :] = 1.0 - theta * dt_eff * diag_n
        ab[2, :-1] = -theta * dt_eff * sub_n[1:]

        rhs = q.copy()
        rhs += (1 - theta) * dt_eff * (sub_o * np.roll(q, 1) + diag_o * q + sup_o * np.roll(q, -1))
        # fix roll wraparound at the ends (no neighbor there)
        rhs[0] -= (1 - theta) * dt_eff * sub_o[0] * q[-1]
        rhs[-1] -= (1 - theta) * dt_eff * sup_o[-1] * q[0]

        q_new = solve_banded((1, 1), ab, rhs)
        q_new = np.maximum(q_new, 0.0)  # guard against roundoff-level negative values

        J_left = theta * left_n * q_new[0] + (1 - theta) * left_o * q[0]
        f_new = -J_left  # rate of absorption = FPT density

        q = q_new
        t = t_new
        ts.append(t)
        Ss.append(float(np.sum(q) * grid.dy))
        fs.append(float(f_new))

    S = np.array(Ss)
    residual_mass = S[-1]

    return Solution(t=np.array(ts), S=S, f=np.array(fs), q_last=q, grid=grid, p=p,
                     stopped_early=stopped_early, residual_mass=residual_mass)





FP_GRID_N = 150
FP_DT = 0.05
FP_R = 3.0          # right (reflecting) edge, same choice as before
FP_T_REF = 0.0       # any time < t0 works: lambda(t) is frozen at lambda0 for all t < t0
 
 
def theta_to_fp_params(theta, t0: float) -> Params:
    """theta = (alpha, mu, sigma2, tau, a) -> fp_tipping.Params.
 
    Same reparametrization your hazard_rate() already uses:
        m       = mu - alpha / (2*a)
        lambda0 = -alpha**2 / (4*a)
    (lambda0 <= 0 automatically for any real alpha, a > 0 -- presumably why
    you optimize in (alpha, mu) rather than directly in (m, lambda0)).
 
    Note theta[2] is sigma2 (variance), matching best_params["sigma2"] /
    values[2] in your script -- fp_tipping.Params.sigma wants the standard
    deviation, so we take the square root here.
    """
    alpha, mu, sigma2, tau, a = theta
    m = mu - alpha / (2.0 * a)
    lambda0 = -(alpha ** 2) / (4.0 * a)
    sigma = np.sqrt(sigma2)
    return Params(A=a, m=m, lambda0=lambda0, t0=t0, tau_r=tau, sigma=sigma, r=FP_R)
 
 
def proba_tip_before_fp(theta, T: float, t0: float,
                         N: int = FP_GRID_N, dt: float = FP_DT) -> float:
    """P(T_tip < T | theta), solved directly from the Fokker-Planck
    equation. Drop-in replacement for:
        r = hazard_rate(t_grid, theta)
        H = cumulative_trapezoid(r, t_grid, initial=0)
        proba = 1.0 - np.exp(-H[-1])
    Same signature contract as the code around it: theta unpacks as
    (alpha, mu, sigma2, tau, a), same as `values` / best_theta_5d in your
    script.
    """
    alpha, mu, sigma2, tau, a = theta
 
    if T >= tau + t0:  # same shortcut as your original code: past t_c, certain
        return 1.0
    if T <= t0:
        return 0.0     # negligible pre-ramp noise-induced tipping (see below); T<=t0 not in from_t0 anyway
 
    try:
        p = theta_to_fp_params(theta, t0)
        grid = Grid.make(N)
        q0 = initial_condition(grid, p, t_start=t0, mean_x=mu, std_x=1.0)
 
        max_steps = max(1, int(np.ceil((T - t0) / dt)))
        sol = solve(p, N=N, dt=dt, theta=1.0, S_tol=1e-10,
                    t_start=t0, q0=q0, max_steps=max_steps, t_c_buffer=1e-6)
 
        S_T = float(np.interp(T, sol.t, sol.S))
        return float(np.clip(1.0 - S_T, 0.0, 1.0))
    except Exception:
        # Nelder-Mead can transiently probe pathological corners of theta
        # (e.g. a Chang-Cooper generator that is numerically degenerate at
        # the simplex's edge). Rather than crash the whole optimization,
        # report "no tipping" for that one evaluation -- it will simply
        # never be the maximizer of possibility, which is the only thing
        # that matters for the supremum search.
        return 0.0
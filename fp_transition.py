"""
Numerical transition densities of the saddle-node ("fold") tipping model

    dX_t = -[A (X_t - m)^2 + lambda(t)] dt + sigma dW_t,

    lambda(t) = lambda0 * (1 - 1{t >= t0} * (t - t0) / tau_r),

obtained by solving, for every observed transition (x_{k-1}, t_{k-1}) -> t_k,
the forward Kolmogorov / Fokker-Planck equation

    p_t = -d_x[ b(x,t) p - (sigma^2/2) p_x ],     p(x, t_{k-1}) = delta(x - x_{k-1}),

with b(x,t) = -(A (x-m)^2 + lambda(t)).

This is an adaptation of fp_tipping.py (first-passage-time solver). Same
Scharfetter-Gummel / Chang-Cooper finite-volume scheme, same theta-method
time stepping, same y in [0,1] grid. What changed, and why:

  1. Boundaries. There is no absorbing threshold at ell(t) any more: the true
     process can cross the unstable equilibrium and come back, so absorbing
     there would bias p. Each transition gets its own FIXED window
     [x_lo, x_hi] (so the Landau map y = (x - x_lo)/(x_hi - x_lo) has zero
     mesh velocity). The window is sized from the linear-noise (LNA) mean and
     variance along the deterministic path, +/- K standard deviations.
     Left edge: absorbing (Dirichlet 0). Mass that leaves there is the part of
     the transition that "escapes" (the true model blows up to -infinity in
     finite time once below the unstable branch); the left edge is never put
     above the LNA window and never below a point of no return x_esc (see
     `x_escape`). Right edge: reflecting, as before (carries ~0 mass).

  2. Initial condition. A Dirac delta cannot be represented on the grid, so
     we regularize it with ONE Euler-Maruyama step of length eps:
         p(x, t_{k-1} + eps) ~= N(x_{k-1} + b(x_{k-1}, t_{k-1}) eps, sigma^2 eps),
     with eps chosen so that this Gaussian is resolved by ~`init_cells`
     cells (and capped at `eps_frac_max` * Delta t). Local error O(eps^2) in
     the mean, O(eps^2) in the variance.

  3. Batching. All n transitions are advanced together. Each has its own
     tridiagonal operator; stacked, they form ONE block-diagonal tridiagonal
     matrix (the coupling entries between blocks are exactly zero), so a
     single `solve_banded` call per time step solves all of them. Each block
     uses its own time t_k(j) and its own step dt_k, so irregular sampling is
     fine.

Output: the transition densities p_theta(. | x_{k-1}) on each block's grid,
plus helpers for
    - p_theta(x_k | x_{k-1})       (the true-model likelihood, see note in
                                    the accompanying tex file), and
    - D_B(p_theta(.|x_{k-1}), q(.|x_{k-1}))  for any approximate q.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.linalg import solve_banded


# --------------------------------------------------------------------------
# Model parameters  (same as fp_tipping.Params minus the FPT-specific fields)
# --------------------------------------------------------------------------

@dataclass
class Params:
    A: float = 1.0
    m: float = 1.0
    lambda0: float = -0.5
    t0: float = 5.0
    tau_r: float = 20.0
    sigma: float = 0.2
    x_esc: Optional[float] = None   # point of no return; None -> x_escape() default

    @property
    def t_c(self) -> float:
        """Bifurcation time: lambda(t_c) = 0, the two fixed points merge."""
        return self.t0 + self.tau_r


def lam(t: np.ndarray, p: Params) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return np.where(t < p.t0, p.lambda0, p.lambda0 * (1.0 - (t - p.t0) / p.tau_r))


def drift(x: np.ndarray, t, p: Params) -> np.ndarray:
    """b(x,t) = -(A (x-m)^2 + lambda(t))."""
    return -(p.A * (x - p.m) ** 2 + lam(t, p))


def drift_x(x: np.ndarray, t, p: Params) -> np.ndarray:
    """d b / d x = -2 A (x - m)."""
    return -2.0 * p.A * (x - p.m)


def x_escape(p: Params, c: float = 5.0) -> float:
    """Point of no return used as a hard lower limit for the windows.

    Below the unstable branch ell(t) = m - sqrt(-lambda/A) the drift is
    ~ -A (x-m)^2. A particle at distance d below the widest unstable branch
    (lambda = lambda0) faces a potential barrier ~ A d^3 / 3 against noise
    sigma^2/2, so for d = c (sigma^2/A)^{1/3} the return probability is
    ~exp(-2 c^3 / 3), i.e. ~1e-36 for c = 5. Absorbing there is exact to any
    practical precision.
    """
    if p.x_esc is not None:
        return float(p.x_esc)
    ell_widest = p.m - np.sqrt(max(-p.lambda0, 0.0) / p.A)
    return float(ell_widest - c * (p.sigma ** 2 / p.A) ** (1.0 / 3.0))


# --------------------------------------------------------------------------
# Scharfetter-Gummel / Chang-Cooper machinery  (unchanged, now vectorized
# over a leading "transition" axis)
# --------------------------------------------------------------------------

def _bernoulli(w: np.ndarray) -> np.ndarray:
    """B(w) = w / (exp(w) - 1), evaluated in a numerically stable way."""
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


def _face_state(t: np.ndarray, p: Params, grid: Grid,
                x_lo: np.ndarray, x_hi: np.ndarray):
    """Face velocities / diffusivity for all blocks at their own times.

    t, x_lo, x_hi have shape (n, 1). The window is fixed in time, so the
    mesh velocity of the y-map is zero (was ell'(t)(1-y) in fp_tipping).
    """
    L = x_hi - x_lo                                   # (n,1)
    D = 0.5 * p.sigma ** 2
    Dloc = D / L ** 2                                 # (n,1)
    x_f = x_lo + L * grid.y_faces                     # (n,N+1)
    v_f = drift(x_f, t, p) / L                        # (n,N+1)
    return L, Dloc, v_f


def assemble_operator(t, p: Params, grid: Grid, x_lo, x_hi):
    """Tridiagonal generator of dq/dt = L(t) q for each block (shape (n,N)),
    q_t = -d_y[ v q - Dloc q_y ], q(0)=0 (absorbing), zero flux at y=1.

    Returns (sub, diag, sup, left_flux_coeff) with the same meaning as in
    fp_tipping.assemble_operator. sub[:,0] = sup[:,-1] = 0, so the stacked
    block-diagonal matrix has no coupling between transitions.
    """
    N = grid.N
    dy = grid.dy
    L, Dloc, v_f = _face_state(t, p, grid, x_lo, x_hi)

    h = np.full(N, dy)
    h[0] = dy / 2.0  # left boundary face sits half a cell from the first center

    v = v_f[:, :N]
    w = v * h / Dloc
    beta = -(Dloc / h) * _bernoulli(w)        # coefficient of q_k in J_k

    alpha = np.zeros_like(v)                  # coefficient of q_{k-1} in J_k
    w_int = v[:, 1:] * dy / Dloc
    alpha[:, 1:] = (Dloc / dy) * _bernoulli(-w_int)

    sub = np.zeros_like(v)
    diag = np.zeros_like(v)
    sup = np.zeros_like(v)

    diag[:, :-1] = (beta[:, :-1] - alpha[:, 1:]) / dy
    sup[:, :-1] = -beta[:, 1:] / dy
    sub[:, 1:] = alpha[:, 1:] / dy
    diag[:, -1] = beta[:, -1] / dy             # right flux is zero: reflecting

    left_flux_coeff = beta[:, 0]               # J_{1/2} = beta0 * q0
    return sub, diag, sup, left_flux_coeff


def _apply(sub, diag, sup, q):
    """(L q) for each block, no wrap-around."""
    out = diag * q
    out[:, 1:] += sub[:, 1:] * q[:, :-1]
    out[:, :-1] += sup[:, :-1] * q[:, 1:]
    return out


# --------------------------------------------------------------------------
# Window selection: linear-noise approximation along the deterministic path
# --------------------------------------------------------------------------

def _lna_path(x0, t0, t1, p: Params, n_sub: int, x_floor: float):
    """RK4 for  xbar' = b(xbar,t),  V' = 2 b_x(xbar,t) V + sigma^2.

    Returns min/max of xbar over the path, max V, and (xbar, V) at t1.
    xbar is clipped at x_floor (finite-time blow-up below the unstable branch).
    """
    s2 = p.sigma ** 2
    h = (t1 - t0) / n_sub

    def rhs(x, V, t):
        return drift(x, t, p), 2.0 * drift_x(x, t, p) * V + s2

    x = np.array(x0, dtype=float)
    V = np.zeros_like(x)
    t = np.array(t0, dtype=float)
    xmin, xmax, Vmax = x.copy(), x.copy(), V.copy()
    for _ in range(n_sub):
        k1x, k1V = rhs(x, V, t)
        k2x, k2V = rhs(x + 0.5 * h * k1x, V + 0.5 * h * k1V, t + 0.5 * h)
        k3x, k3V = rhs(x + 0.5 * h * k2x, V + 0.5 * h * k2V, t + 0.5 * h)
        k4x, k4V = rhs(x + h * k3x, V + h * k3V, t + h)
        x = x + h / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
        V = V + h / 6.0 * (k1V + 2 * k2V + 2 * k3V + k4V)
        t = t + h
        x = np.maximum(x, x_floor)
        V = np.clip(np.nan_to_num(V, nan=1e30, posinf=1e30), 0.0, 1e30)
        xmin = np.minimum(xmin, x)
        xmax = np.maximum(xmax, x)
        Vmax = np.maximum(Vmax, V)
    return xmin, xmax, Vmax, x, V


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------

@dataclass
class TransitionSolution:
    x: np.ndarray          # (n, N) cell centers of each block's window
    pdf_grid: np.ndarray   # (n, N) p_theta(x | x_{k-1}) on those centers
    dx: np.ndarray         # (n,)  cell width
    x_lo: np.ndarray       # (n,)  window
    x_hi: np.ndarray       # (n,)
    mass: np.ndarray       # (n,)  int p dx  (= 1 - escaped, up to roundoff)
    escaped: np.ndarray    # (n,)  mass absorbed at x_lo during the transition
    eps: np.ndarray        # (n,)  length of the regularizing Euler-Maruyama step
    init_cells: np.ndarray # (n,)  cells per std of the regularized delta (resolution check)
    x_prev: np.ndarray
    t_prev: np.ndarray
    t_next: np.ndarray
    p: Params

    def pdf(self, xq: np.ndarray) -> np.ndarray:
        """p_theta(xq_k | x_{k-1}) for one query point per transition,
        by linear interpolation on the uniform cell-center grid (0 outside)."""
        xq = np.asarray(xq, dtype=float)
        n, N = self.pdf_grid.shape
        s = (xq - self.x[:, 0]) / self.dx
        i = np.floor(s).astype(int)
        inside = (s >= 0) & (i < N - 1)
        i = np.clip(i, 0, N - 2)
        w = s - i
        rows = np.arange(n)
        val = (1 - w) * self.pdf_grid[rows, i] + w * self.pdf_grid[rows, i + 1]
        return np.where(inside, np.maximum(val, 0.0), 0.0)

    def loglik(self, x_next: np.ndarray, floor: float = 1e-300) -> float:
        """sum_k log p_theta(x_k | x_{k-1}): the true-model log-likelihood."""
        return float(np.sum(np.log(np.maximum(self.pdf(x_next), floor))))

    def bhattacharyya(self, q_pdf: Callable[[np.ndarray], np.ndarray],
                      q_escaped: Optional[np.ndarray] = None) -> np.ndarray:
        """D_B(p_theta(.|x_{k-1}), q(.|x_{k-1})) for every transition.

        q_pdf(x) must accept the (n, N) array self.x and return q(x | x_{k-1})
        row by row (row k conditioned on x_prev[k], over [t_prev[k], t_next[k]]).

        The true transition is DEFECTIVE: mass `escaped` has gone past the point
        of no return. It is treated as an atom on a cemetery state, so that
            BC = int sqrt(p q) dx + sqrt(p_esc * q_esc)
        and D_B(p, p) = 0 exactly (without the atom, D_B(p,p) = -log(1-p_esc)).
        q_escaped: the approximate model's mass beyond the point of no return.
        Default: q's mass NOT on the window, 1 - int_window q dx (the right tail
        beyond x_hi is ~0 by construction of the window, so this is essentially
        q's left tail).
        """
        q = np.maximum(np.asarray(q_pdf(self.x), dtype=float), 0.0)
        if q_escaped is None:
            q_escaped = np.clip(1.0 - np.sum(q, axis=1) * self.dx, 0.0, 1.0)
        p_esc = np.clip(self.escaped, 0.0, 1.0)
        bc = np.sum(np.sqrt(np.maximum(self.pdf_grid, 0.0) * q), axis=1) * self.dx
        bc = bc + np.sqrt(p_esc * np.asarray(q_escaped, dtype=float))
        return np.maximum(-np.log(np.clip(bc, 1e-300, 1.0)), 0.0)


def transition_densities(p: Params, x_prev, t_prev, t_next,
                         N: int = 200, n_steps: int = 100, K: float = 8.0,
                         theta: float = 0.5, rannacher_steps: int = 4,
                         init_cells: float = 4.0, eps_frac_max: float = 0.05,
                         ) -> TransitionSolution:
    """Solve the Fokker-Planck equation for all transitions x_prev[k] at
    t_prev[k]  ->  t_next[k] at once.

    N               cells per transition window.
    n_steps         time steps per transition (each block uses its own dt).
    K               window half-width in LNA standard deviations.
    theta           1.0 backward Euler, 0.5 Crank-Nicolson (default).
    rannacher_steps with theta<1, the first steps are done in backward Euler
                    (half-size, Rannacher start-up) to damp the CN
                    oscillations from the sharp initial Gaussian.
    init_cells      the regularized delta has std = init_cells * dx.
    eps_frac_max    cap eps <= eps_frac_max * (t_next - t_prev).
    """
    x_prev = np.atleast_1d(np.asarray(x_prev, dtype=float))
    t_prev = np.broadcast_to(np.asarray(t_prev, dtype=float), x_prev.shape).copy()
    t_next = np.broadcast_to(np.asarray(t_next, dtype=float), x_prev.shape).copy()
    if not (np.all(np.isfinite(x_prev)) and np.all(np.isfinite(t_prev)) and np.all(np.isfinite(t_next))):
        raise ValueError("non-finite x_prev / t_prev / t_next")
    if np.any(t_next <= t_prev):
        raise ValueError("t_next must be > t_prev for every transition")
    n = x_prev.size
    grid = Grid.make(N)
    Dt = t_next - t_prev

    # ---- windows ---------------------------------------------------------
    x_floor = min(x_escape(p), float(np.min(x_prev)) - 1.0)
    xmin, xmax, Vmax, _, _ = _lna_path(x_prev, t_prev, t_next, p, n_sub=64, x_floor=x_floor)
    sd = np.sqrt(np.maximum(Vmax, 1e-4 * p.sigma ** 2 * Dt))
    x_lo = np.maximum(xmin - K * sd, x_floor)
    x_hi = xmax + K * sd
    Lw = x_hi - x_lo
    dx = Lw * grid.dy

    # ---- regularized delta: one Euler-Maruyama step of length eps ---------
    eps = np.minimum((init_cells * dx / p.sigma) ** 2, eps_frac_max * Dt)
    mu0 = x_prev + drift(x_prev, t_prev, p) * eps
    s0 = p.sigma * np.sqrt(eps)
    xc = x_lo[:, None] + Lw[:, None] * grid.y_centers[None, :]
    q = np.exp(-0.5 * ((xc - mu0[:, None]) / s0[:, None]) ** 2)
    q /= np.sum(q, axis=1, keepdims=True) * grid.dy   # q = p * L, mass 1 in y

    # ---- time stepping ----------------------------------------------------
    t_start = t_prev + eps
    dt_blk = (t_next - t_start) / n_steps

    # (sub-step schedule: Rannacher start = 2*R half steps of BE, then theta)
    schedule = []
    R = rannacher_steps if theta < 1.0 else 0
    for _ in range(R):
        schedule += [(0.5, 1.0), (0.5, 1.0)]
    schedule += [(1.0, theta)] * (n_steps - R)

    tl = t_start[:, None]
    xl = x_lo[:, None]
    xh = x_hi[:, None]
    M = n * N
    escaped = np.zeros(n)
    ops_old = assemble_operator(tl, p, grid, xl, xh)

    for frac, th in schedule:
        dt_eff = (frac * dt_blk)[:, None]         # (n,1)
        t_new = tl + dt_eff
        sub_n, diag_n, sup_n, left_n = assemble_operator(t_new, p, grid, xl, xh)
        sub_o, diag_o, sup_o, left_o = ops_old

        # (I - th*dt*L_new) q_new = (I + (1-th)*dt*L_old) q_old
        ab = np.zeros((3, M))
        ab[0, 1:] = (-th * dt_eff * sup_n).ravel()[:-1]
        ab[1, :] = (1.0 - th * dt_eff * diag_n).ravel()
        ab[2, :-1] = (-th * dt_eff * sub_n).ravel()[1:]

        rhs = q.copy()
        if th < 1.0:
            rhs += (1 - th) * dt_eff * _apply(sub_o, diag_o, sup_o, q)

        q_new = solve_banded((1, 1), ab, rhs.ravel()).reshape(n, N)
        q_new = np.maximum(q_new, 0.0)

        J_left = th * left_n * q_new[:, 0] + (1 - th) * left_o * q[:, 0]
        escaped += -J_left * dt_eff[:, 0]

        q = q_new
        tl = t_new
        ops_old = (sub_n, diag_n, sup_n, left_n)

    pdf_grid = q / Lw[:, None]
    mass = np.sum(q, axis=1) * grid.dy
    return TransitionSolution(x=xc, pdf_grid=pdf_grid, dx=dx, x_lo=x_lo, x_hi=x_hi,
                              mass=mass, escaped=escaped, eps=eps,
                              init_cells=s0 / dx, x_prev=x_prev, t_prev=t_prev,
                              t_next=t_next, p=p)


# --------------------------------------------------------------------------
# theta = (alpha, mu, sigma2, tau, a)  ->  Params   (same map as fp_tipping)
# --------------------------------------------------------------------------

def theta_to_fp_params(theta, t0: float) -> Params:
    alpha, mu, sigma2, tau, a = theta
    m = mu - alpha / (2.0 * a)
    lambda0 = -(alpha ** 2) / (4.0 * a)
    return Params(A=a, m=m, lambda0=lambda0, t0=t0, tau_r=tau, sigma=np.sqrt(sigma2))


def transition_densities_theta(theta, x_obs, t_obs, t0: float, **kw) -> TransitionSolution:
    """All n-1 transitions of an observed path (x_obs[k], t_obs[k]), k=0..n-1."""
    x_obs = np.asarray(x_obs, dtype=float)
    t_obs = np.asarray(t_obs, dtype=float)
    p = theta_to_fp_params(theta, t0)
    return transition_densities(p, x_obs[:-1], t_obs[:-1], t_obs[1:], **kw)


# --------------------------------------------------------------------------
# Example q: linear-noise (LNA) Gaussian. Only for testing -- plug your own
# linearized / Strang transition density in its place.
# --------------------------------------------------------------------------

def lna_gaussian_pdf(p: Params, x_prev, t_prev, t_next, n_sub: int = 200):
    x_prev = np.atleast_1d(np.asarray(x_prev, dtype=float))
    x_floor = min(x_escape(p), float(np.min(x_prev)) - 1.0)
    _, _, _, xT, VT = _lna_path(x_prev, np.asarray(t_prev, float),
                                np.asarray(t_next, float), p, n_sub, x_floor)

    def q(x):
        s2 = VT[:, None]
        return np.exp(-0.5 * (x - xT[:, None]) ** 2 / s2) / np.sqrt(2 * np.pi * s2)
    return q


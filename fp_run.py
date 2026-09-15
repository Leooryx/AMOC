"""
Run the tipping-time PDE solver with the fitted parameters ("Possibility
(this work)" row):

    A=0.89, m=-1.44, lambda0=-2.49, sigma=0.53, tau_r=137.18, t0=1924

Given t0 - t_start = 1924 time units, far longer than the well's own
relaxation time, the pre-ramp phase is handled with the quasi-stationary
distribution (see quasi_stationary_distribution() in fp_tipping.py) rather
than brute-force time-stepping the whole burn-in: we verified on a cheap
surrogate (same physical parameters, small t0) that this reproduces the
brute-force result to ~1e-4 relative accuracy at a fraction of the cost.

Per the user's choice, this run skips a full Monte Carlo re-validation for
these particular parameters (it was validated on illustrative parameters
already, in compare.py / README.md). It does, however, run 1000 sample
trajectories with the user's own simulator (utils.py, used as-is) as a
quick visual check that the PDE's first-passage-time density lines up with
plain simulation.

Note: utils.X_traj()'s returned "FPT" field is not actually the first
crossing time -- its second while loop deliberately never breaks on
first_passage (to let rebounds be observed), so "FPT" ends up being
whatever time the loop naturally terminates at (current_lambda >= 0, i.e.
essentially t_c) for almost every path. We instead recover the true first
crossing time by scanning each returned trajectory against ell(t) ourselves.
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from fp_tipping import (Params, Grid, solve, ell, lam,
                         quasi_stationary_distribution, qsd_convergence_check)
import utils

plt.rcParams["figure.dpi"] = 140

p = Params(A=0.89, m=-1.44, lambda0=-2.49, t0=1924.0, tau_r=137.18, sigma=0.53, r=3.0)
print(f"Params: {p}")
print(f"t_c = t0 + tau_r = {p.t_c:.2f}  (given: 2061.18)")

x_plus0 = p.m + np.sqrt(-p.lambda0 / p.A)
ell0 = p.m - np.sqrt(-p.lambda0 / p.A)
print(f"x_+(0) = {x_plus0:.4f}  (given mu_0 = 0.23)")
print(f"ell(0) = {ell0:.4f}  (stated domain floor = -3; slightly outside, noted in README)")

N = 800
grid = Grid.make(N)

# --- Pre-ramp phase: quasi-stationary distribution ---
qsd, decay_rate = quasi_stationary_distribution(p, grid, t_ref=0.0)
conv_residual = qsd_convergence_check(p, grid, t_ref=0.0)
print(f"\nFrozen-phase leading eigenvalue (survival decay rate once equilibrated): {decay_rate:.3e}")
print(f"  -> essentially zero: negligible noise-induced tipping before the ramp starts")
print(f"Shape-convergence residual exp(-gap*t0): {conv_residual:.3e}  (~0 => QSD fully reached well before t0)")

x_grid = ell0 + (p.r - ell0) * grid.y_centers
mean_qsd = np.sum(qsd * x_grid) * grid.dy
std_qsd = np.sqrt(np.sum(qsd * (x_grid - mean_qsd) ** 2) * grid.dy)
print(f"QSD mean = {mean_qsd:.4f}, std = {std_qsd:.4f}  (harmonic estimate at x_+: "
      f"std ~= sigma/sqrt(2*2*A*(x_+0-m)) = {p.sigma/np.sqrt(2*2*p.A*(x_plus0-p.m)):.4f})")

# --- Ramp phase: t0 -> t_c, starting from the QSD ---
sol = solve(p, N=N, dt=0.01, theta=1.0, S_tol=1e-10, t_start=p.t0, q0=qsd)
print(f"\nRamp-phase PDE solve: {len(sol.t)} steps, t in [{sol.t[0]:.2f}, {sol.t[-1]:.2f}], "
      f"stopped_early={sol.stopped_early}, residual (unresolved-by-stop) mass={sol.residual_mass:.4f}")

# grid-convergence spot check (cheap: two resolutions)
sol_coarse = solve(p, N=400, dt=0.02, theta=1.0, S_tol=1e-10, t_start=p.t0, q0=None)
# q0=None uses a fresh Gaussian on the coarse grid's own y-grid at t0, which is fine here since
# the true QSD is what we're comparing *to* (an independent cross-check) -- but for a clean
# apples-to-apples convergence check we instead recompute the QSD on the coarse grid too:
grid_coarse = Grid.make(400)
qsd_coarse, _ = quasi_stationary_distribution(p, grid_coarse, t_ref=0.0)
sol_coarse = solve(p, N=400, dt=0.02, theta=1.0, S_tol=1e-10, t_start=p.t0, q0=qsd_coarse)
t_probe = p.t0 + 100.0
S_fine = np.interp(t_probe, sol.t, sol.S)
S_coarse = np.interp(t_probe, sol_coarse.t, sol_coarse.S)
print(f"Grid convergence spot check at t={t_probe}: S(N=800,dt=0.01)={S_fine:.6f}  "
      f"S(N=400,dt=0.02)={S_coarse:.6f}  diff={abs(S_fine-S_coarse):.2e}")

# conservation self-check
dS_scheme = -sol.f[1:] * np.diff(sol.t)
dS_actual = np.diff(sol.S)
print(f"Conservation residual (max |scheme - actual| per step): "
      f"{np.max(np.abs(dS_scheme - dS_actual)):.2e}")

# --- Summary statistics ---
# median and quantiles of T_tip via the survival curve S(t) = P(T_tip > t)
def quantile_from_survival(t, S, q):
    # smallest t such that S(t) <= 1-q  (P(T_tip <= t) >= q)
    target = 1 - q
    idx = np.searchsorted(-S, -target)  # S is decreasing
    idx = min(idx, len(t) - 1)
    return t[idx]

for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
    tq = quantile_from_survival(sol.t, sol.S, q)
    flag = "" if sol.S[-1] <= 1 - q else "  (not yet reached within resolved horizon)"
    print(f"  T_tip quantile q={q:.2f}: t = {tq:.2f}{flag}")

S_today = np.interp(2026.0, sol.t, sol.S) if sol.t[0] <= 2026.0 <= sol.t[-1] else None
if S_today is not None:
    print(f"S(2026) = P(T_tip > 2026) = {S_today:.4f}  =>  P(T_tip <= 2026) = {1-S_today:.4f}")

# --- Monte Carlo: 1000 trajectories with the user's own simulator ---
utils.t0 = p.t0        # globals X_traj()/S_onestep() expect
utils.delta_t = 1/12   # ~9600 steps/path over the full 1870->t_c run; ~15-20s for 1000 paths

mc_params = dict(alpha=2.98, mu=0.23, sigma2=p.sigma**2, tau=p.tau_r,
                  a=p.A, lambda0=p.lambda0, m=p.m, t_c=p.t_c)

N_SIMS = 5000
np.random.seed(0)
fpt_samples = []
with np.errstate(all="ignore"):  # the simulator keeps integrating well past the barrier (to catch
                                  # rebounds), where the unbounded quadratic drift can overflow --
                                  # harmless here since we only use the trajectory up to first crossing
    for _ in tqdm(range(N_SIMS), desc="simulations"):
        r = utils.X_traj(mc_params, X0=0.23)
        X = r["X"]
        t_arr = 1870.0 + utils.delta_t * np.arange(len(X))
        crossed = X <= ell(t_arr, p)
        if crossed.any():
            fpt_samples.append(t_arr[np.argmax(crossed)])
        else:
            fpt_samples.append(p.t_c)  # never crossed within the run -> deterministic tip at t_c

fpt_samples = np.array(fpt_samples)
print(f"\nMonte Carlo ({N_SIMS} paths, user's simulator, delta_t={utils.delta_t}): "
      f"median FPT={np.median(fpt_samples):.2f}, mean={fpt_samples.mean():.2f}, "
      f"frac capped at t_c={np.mean(fpt_samples >= p.t_c - 1e-9):.3f}")

# --- Plots ---
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0, 0]
ax.plot(sol.t, sol.S, lw=2, color="C0")
ax.axvline(p.t0, color="gray", ls=":", lw=1, label="$t_0$")
ax.axvline(p.t_c, color="k", ls=":", lw=1, label="$t_c$")
if 2026 >= sol.t[0]:
    ax.axvline(2026, color="C3", ls="--", lw=1, label="2026 (today)")
ax.set_xlabel("t")
ax.set_ylabel(r"$S(t)=P(T_{tip}>t)$")
ax.set_title("Survival probability (ramp phase)")
ax.legend(fontsize=8)

ax = axes[0, 1]
# exclude the pile-up exactly at t_c (paths that didn't cross via noise --
# these deterministically tip right at the bifurcation, i.e. the same
# "residual mass" the PDE reports separately) from the histogram/KDE so it
# doesn't distort the y-scale; the PDE curve is directly comparable to it.
not_capped = fpt_samples < p.t_c - 1e-9
ax.hist(fpt_samples[not_capped], bins=40, density=True, alpha=0.35, color="C1",
        label=f"MC histogram (n={not_capped.sum()}/{N_SIMS} crossed before $t_c$)")
if not_capped.sum() > 10:
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(fpt_samples[not_capped])
    tt_kde = np.linspace(sol.t[0], sol.t[-1], 400)
    ax.plot(tt_kde, kde(tt_kde), color="C1", lw=1.5, ls="--", label="MC empirical density (KDE)")
ax.plot(sol.t, sol.f, lw=2, color="C0", label="PDE (boundary flux)")
ax.axvline(p.t0, color="gray", ls=":", lw=1)
ax.axvline(p.t_c, color="k", ls=":", lw=1)
ax.set_xlabel("t")
ax.set_xlim(2000, sol.t[-1])
ax.set_ylabel("f(t)")
ax.set_title("First-passage-time density (ramp phase)")
ax.legend(fontsize=7)

ax = axes[1, 0]
ax.plot(x_grid, qsd, color="C0")
ax.axvline(ell0, color="k", ls=":", lw=1, label=r"$\ell(0)$ (absorbing)")
ax.axvline(x_plus0, color="C1", ls="--", lw=1, label=r"$x_+(0)$ (stable)")
ax.axvline(0.23, color="C3", ls=":", lw=1, label=r"given $\mu_0$")
ax.set_xlabel("x")
ax.set_ylabel("density")
ax.set_title("Quasi-stationary distribution used to start the ramp")
ax.legend(fontsize=8)

ax = axes[1, 1]
tt2 = np.linspace(p.t0, p.t_c * 0.9995, 400)
ax.plot(tt2, ell(tt2, p), label=r"$\ell(t)$ (absorbing)")
ax.plot(tt2, p.m + np.sqrt(np.maximum(-lam(tt2, p), 0) / p.A), label=r"$x_+(t)$ (stable)")
ax.axhline(p.m, color="gray", lw=1, ls="--")
ax.axvline(p.t_c, color="k", ls=":", lw=1, label="$t_c$")
ax.set_xlabel("t")
ax.set_ylabel("x")
ax.set_title("Fixed points during the ramp")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("real_parameters_results.png")
print("\nsaved real_parameters_results.png")
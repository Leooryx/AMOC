"""
Post-processing of the files written by run_experiments.py.

Per experiment (written in the experiment folder `<exp>/`):
    <exp>_hist_box<size>.png             increase of credibility / necessity per bin
    <exp>_credibility_intervals.txt      95% credibility interval of every parameter (+ t_c)

Across experiments (written in `outputs/`):
    Baseline_vs_Widened_possTip.png      cumulative credibility / necessity (Widened dotted)
    <exp>_MC_trajectories.png            Monte Carlo trajectories for several parameter sets
    <exp>_MC_param_sets.txt              parameter sets, possibility values and colors used
"""

import os
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from utils import *
from likelihoods import *
from constants import *


EXPERIMENT_NAMES = ["Baseline", "Widened", "FP_transition"]
params = ["alpha", "mu", "sigma2", "tau", "a"]
ORIGIN_YEAR = 1924          # t_c = tau + ORIGIN_YEAR
CI_LEVEL = 0.95             # credibility level of the intervals
OUTPUTS_DIR = "outputs"


# =============================================================================
# Paths of the files written by run_experiments.py
# =============================================================================

def experiment_paths(exp_name):
    p = lambda suffix: os.path.join(exp_name, f"{exp_name}_{suffix}")
    return {
        "dir": exp_name,
        "mle": p("mle.json"),
        "profiles": p("marginal_profiles.txt"),
        "poss_nec": p("poss_nec.txt"),
    }


def experiment_is_complete(exp_name):
    paths = experiment_paths(exp_name)
    return all(os.path.exists(paths[k]) for k in ("mle", "profiles", "poss_nec"))


# =============================================================================
# Loaders
# =============================================================================

def load_mle(exp_name):
    with open(experiment_paths(exp_name)["mle"]) as f:
        return json.load(f)


def load_poss_nec(exp_name):
    """Returns a dict: T, poss, nec, <param>, marg_<param>."""
    data = np.loadtxt(experiment_paths(exp_name)["poss_nec"], comments="#", ndmin=2)
    cols = ["T", "poss", "nec"] + params + [f"marg_{p}" for p in params]
    return {c: data[:, k] for k, c in enumerate(cols)}


def load_profiles(exp_name):
    """Returns (profiles, grids), two dicts indexed by parameter name."""
    grids = {p: [] for p in params}
    profiles = {p: [] for p in params}
    with open(experiment_paths(exp_name)["profiles"]) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            p, x, y = line.split()
            grids[p].append(float(x))
            profiles[p].append(float(y))
    order = {p: np.argsort(grids[p]) for p in params}
    grids = {p: np.asarray(grids[p])[order[p]] for p in params}
    profiles = {p: np.asarray(profiles[p])[order[p]] for p in params}
    return profiles, grids


# =============================================================================
# 1) Histograms of credibility / necessity
# =============================================================================

def plot_histograms(exp_name, box_size, t_c=None):
    """Bars of width `box_size` (years). The height of a bar is the increase of the
    cumulative curve P(T_tip < t) (resp. N(T_tip < t)) over that bin, read off the
    saved curve by linear interpolation.
    Note: the curve is saved on 50 points only, so a box smaller than the grid
    spacing does not bring extra information."""
    d = load_poss_nec(exp_name)
    T = d["T"]

    edges = np.append(np.arange(T[0], T[-1], box_size), T[-1])
    widths = np.diff(edges)
    d_poss = np.diff(np.interp(edges, T, d["poss"]))
    d_nec = np.diff(np.interp(edges, T, d["nec"]))

    plt.figure(figsize=(8, 5))
    plt.bar(edges[:-1], d_poss, width=widths, align="edge", color="blue", alpha=0.35,
            edgecolor="blue", linewidth=0.6, label=r"Credibility $\bar{P}$")
    plt.bar(edges[:-1], d_nec, width=widths, align="edge", color="yellow", alpha=0.5,
            edgecolor="goldenrod", linewidth=0.6, label=r"Necessity $N$")
    if t_c is not None:
        plt.axvline(t_c, color="red", linestyle="--", alpha=0.7, label=f"$t_c$ from MLE (~ {t_c:.1f})")
    plt.xlabel("Year")
    plt.ylabel(f"Increase over {box_size:g} years")
    plt.title(f"{exp_name}: tipping credibility / necessity per {box_size:g}-year bin")
    plt.grid(alpha=0.3)
    plt.legend()

    path = os.path.join(exp_name, f"{exp_name}_hist_box{box_size:g}.png")
    plt.savefig(path)
    plt.close()
    return path


# =============================================================================
# 2) Credibility intervals
# =============================================================================

def credibility_interval(grid, profile, level=CI_LEVEL):
    """alpha-cut {theta : pi(theta) >= 1 - level} of a marginal possibility
    function (its necessity is >= level). The two bounds are found by linear
    interpolation between the grid points around each crossing.
    Returns (lower, upper, lower_truncated, upper_truncated): a bound is
    'truncated' when pi is still above the threshold at the edge of the grid."""
    cut = 1.0 - level
    above = np.flatnonzero(profile >= cut)
    if above.size == 0:
        return np.nan, np.nan, False, False

    i0, i1 = above[0], above[-1]

    if i0 == 0:
        lower, lower_trunc = grid[0], True
    else:
        lower = np.interp(cut, [profile[i0 - 1], profile[i0]], [grid[i0 - 1], grid[i0]])
        lower_trunc = False

    if i1 == len(grid) - 1:
        upper, upper_trunc = grid[-1], True
    else:  # profile decreasing there: reverse so that np.interp gets increasing x
        upper = np.interp(cut, [profile[i1 + 1], profile[i1]], [grid[i1 + 1], grid[i1]])
        upper_trunc = False

    return lower, upper, lower_trunc, upper_trunc


def write_credibility_intervals(exp_name, level=CI_LEVEL):
    profiles, grids = load_profiles(exp_name)
    best = load_mle(exp_name)["best_params"]

    rows = []
    for p in params:
        lo, up, lo_t, up_t = credibility_interval(grids[p], profiles[p], level)
        rows.append((p, best[p], lo, up, lo_t, up_t))

    # t_c = tau + ORIGIN_YEAR
    _, tau_mle, lo, up, lo_t, up_t = rows[params.index("tau")]
    rows.append(("t_c", tau_mle + ORIGIN_YEAR, lo + ORIGIN_YEAR, up + ORIGIN_YEAR, lo_t, up_t))

    path = os.path.join(exp_name, f"{exp_name}_credibility_intervals.txt")
    with open(path, "w") as f:
        f.write(f"# {exp_name}: {level:.0%} credibility intervals "
                f"(alpha-cut of the marginal possibility at {1 - level:.2f})\n")
        f.write(f"# t_c = tau + {ORIGIN_YEAR}\n")
        f.write("# '*' = the possibility is still above the threshold at the edge of the grid:\n")
        f.write("#       the true bound lies beyond the grid (widen intervals[p] in constants.py)\n")
        f.write("# param\tMLE\tlower\tupper\n")
        for p, mle, lo, up, lo_t, up_t in rows:
            f.write(f"{p}\t{mle:.6g}\t{lo:.6g}{'*' if lo_t else ''}\t{up:.6g}{'*' if up_t else ''}\n")
    return path


def analyse_experiment(exp_name, box_sizes):
    if not experiment_is_complete(exp_name):
        print(f"[{exp_name}] files missing, skipped.")
        return
    best = load_mle(exp_name)["best_params"]
    t_c = best["tau"] + ORIGIN_YEAR
    for box in box_sizes:
        print(f"[{exp_name}] wrote {plot_histograms(exp_name, box, t_c)}")
    print(f"[{exp_name}] wrote {write_credibility_intervals(exp_name)}")


# =============================================================================
# outputs/ 1) Cumulative credibility / necessity of several experiments
# =============================================================================

def plot_compare_possTip(exp_names=("Baseline", "Widened"), linestyles=("-", ":")):
    plt.figure(figsize=(8, 5))
    for exp_name, ls in zip(exp_names, linestyles):
        if not experiment_is_complete(exp_name):
            print(f"[{exp_name}] files missing, not in the comparison plot.")
            continue
        d = load_poss_nec(exp_name)
        t_c = load_mle(exp_name)["best_params"]["tau"] + ORIGIN_YEAR
        plt.plot(d["T"], d["poss"], ls, lw=2, color="blue", label=rf"{exp_name}: credibility $\bar{{P}}$")
        plt.plot(d["T"], d["nec"], ls, lw=2, color="orange", label=rf"{exp_name}: necessity $N$")
        plt.axvline(t_c, color="red", linestyle=ls, alpha=0.7, label=f"{exp_name}: $t_c$ (~ {t_c:.1f})")
    plt.xlabel("Year")
    plt.ylabel("Credibility / Necessity")
    plt.title("OMP of observing AMOC collapse before time t")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)

    path = os.path.join(OUTPUTS_DIR, f"{'_vs_'.join(exp_names)}_possTip.png")
    plt.savefig(path)
    plt.close()
    return path


def plot_compare_marginals(exp_names=("Baseline", "Widened"), linestyles=("-", ":"),
                           mle_colors=("black", "grey"), text_heights=(0.4, 0.75)):
    """Marginal possibility functions of several experiments on the same graph, in
    the style of plot_profiles: one panel per parameter (same colors), one line
    style per experiment (Widened dotted), MLE of each experiment as a vertical line."""
    n_params = len(params)
    labels = ['alpha', 'mu', 'sigma2', 'tau', 'a']
    default_colors = ['#204878', '#3870b0', '#6098d8', '#cc4c02', '#ec7014', '#8c510a']
    colors = default_colors[:n_params]
 
    fig, axes = plt.subplots(1, n_params, figsize=(3.6 * n_params, 4), sharey=True)
 
    for exp_name, ls, mle_c, h in zip(exp_names, linestyles, mle_colors, text_heights):
        if not experiment_is_complete(exp_name):
            print(f"[{exp_name}] files missing, not in the marginals plot.")
            continue
        profiles, grids = load_profiles(exp_name)
        estimates = load_mle(exp_name)["best_params"]
 
        for idx, key in enumerate(params):
            ax = axes[idx]
            ax.plot(grids[key], profiles[key], color=colors[idx], linestyle=ls, linewidth=2.5,
                    label=rf'$\pi(\theta|X)$ {exp_name}')
 
            x = estimates[key]
            ax.axvline(x, color=mle_c, linestyle=':', label=f'MLE {exp_name}')
            ax.text(x, h, f"{x:.3g}", rotation=90, ha='left', va='top', fontsize=9, color=mle_c)
 
    for idx, key in enumerate(params):
        ax = axes[idx]
        ax.set_title(labels[idx], fontsize=14, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=':', alpha=0.5)
 
        if idx == 0:
            ax.set_ylabel('Possibility ', fontsize=12)
            ax.legend(loc='upper right', fontsize=8)
 
    plt.suptitle(f"Marginalized possibilities: {' vs '.join(exp_names)}", fontsize=10, fontweight='bold')
    plt.tight_layout()
 
    path = os.path.join(OUTPUTS_DIR, f"{'_vs_'.join(exp_names)}_marginals.png")
    plt.savefig(path)
    plt.close(fig)
    return path


# =============================================================================
# outputs/ 2) Monte Carlo trajectories for several parameter sets
# =============================================================================



def joint_possibility(theta, likelihood, max_loglik):
    """exp(-likelihood(theta) - max_loglik): possibility of the parameter set."""
    return float(np.clip(np.exp(-likelihood(list(theta)) - max_loglik), 0.0, 1.0))


def plot_monte_carlo(param_sets, num_sim=3, exp_name="Baseline", seed=None,
                     alpha_min=0.1, alpha_max=0.6):
    """`param_sets`: list of [alpha, mu, sigma2, tau, a] (lists or dicts).
    Each set gets one random color (the same for all its n_sim trajectories) and
    a transparency that grows with its joint possibility under `exp_name`'s
    likelihood: alpha = alpha_min + (alpha_max - alpha_min) * possibility.
    Parameters other than the 5 above (lambda0, m) are taken from the MLE of
    `exp_name`; t_c = tau + ORIGIN_YEAR."""
    from run_experiments import EXPERIMENTS   # heavy imports only when needed
    from constants import delta_t, t0
    from utils import X_traj
 
    rng = np.random.default_rng(seed)
    if seed is not None:
        np.random.seed(seed)  # X0 and X_traj use np.random
 
    likelihood = EXPERIMENTS[exp_name]
    mle = load_mle(exp_name)
    max_loglik = mle["max_loglik_global"]
 
    plt.figure(figsize=(10, 4))
    plt.ylim(-3,1)
    summary = []
 
    for k, s in enumerate(param_sets):
        theta = [s[p] for p in params] if isinstance(s, dict) else list(s)
        poss = joint_possibility(theta, likelihood, max_loglik)
        COLORS = ["orange", "pink", "black", "black"] 
        color = to_rgb(COLORS[k % len(COLORS)])
        line_alpha = 0.5 #alpha_min + (alpha_max - alpha_min) * poss
 
        p_full = {**mle["other_params"], **dict(zip(params, theta))}
        p_full["t_c"] = p_full["tau"] + ORIGIN_YEAR
        #barrier = p_full["m"] + np.sqrt(-p_full["lambda0"] / p_full["a"])  # stationary-phase barrier
 
        fpts = []
        for j in range(num_sim):
            # we start not far from the stable position
            X0 = p_full["m"] + np.sqrt(-p_full["lambda0"] / p_full["a"]) + np.random.normal(0, 1)
            sim_result = X_traj(p_full, X0)
            fpts.append(sim_result["FPT"])
 
            time_axis = 1870 + np.arange(len(sim_result["X"])) * delta_t
            plt.plot(time_axis, sim_result["X"], color=color, alpha=line_alpha,
                     label=f"Simulated AMOC paths, set {k + 1} (possibility = {poss:.3f})" if j == 0 else None)
 
        mean_fpt = float(np.mean(fpts))
        print(f"[{exp_name}] set {k + 1}: mean tipping time: {mean_fpt:.2f} years")
        summary.append((k + 1, theta, poss, color, mean_fpt))
 
    #plt.axvline(x=t0, color="red", linestyle="--", label="Ramping starts ($T_0$)")
 
    # Barrier (tipping threshold): one black line if all sets share it, otherwise one per set in its color
    #barriers = [row[4] for row in summary]
    """if np.allclose(barriers, barriers[0]):
        plt.axhline(y=barriers[0], color="black", linestyle=":", label="Barrier (Tipping Threshold)")
    else:
        for k, _, _, color, barrier, _ in summary:
            plt.axhline(y=barrier, color=color, linestyle=":", label=f"Barrier (Tipping Threshold), set {k}")"""
 
    plt.xlabel("Time")
    plt.ylabel("$X_t$ State")
    plt.title(f"AMOC trajectory simulation ({exp_name}, {num_sim} paths per set)")
    leg = plt.legend()
    for h in leg.legend_handles:   # legend lines fully opaque
        h.set_alpha(1)
    plt.grid(True, alpha=0.3)
 
    fig_path = os.path.join(OUTPUTS_DIR, f"{exp_name}_MC_trajectories.png")
    plt.savefig(fig_path)
    plt.close()
 
    txt_path = os.path.join(OUTPUTS_DIR, f"{exp_name}_MC_param_sets.txt")
    with open(txt_path, "w") as f:
        f.write(f"# n_sim = {num_sim}, seed = {seed}, likelihood of {exp_name}\n")
        f.write("# set\t" + "\t".join(params) + "\tpossibility\tmedian_FPT\tcolor_rgb\n")
        for k, theta, poss, color, median_fpt in summary:
            f.write(f"{k}\t" + "\t".join(f"{v:.6g}" for v in theta)
                    + f"\t{poss:.6g}\t{median_fpt:.2f}\t"
                    + ",".join(f"{c:.3f}" for c in color) + "\n")
    return fig_path, txt_path


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    BOX_SIZES = [5]   # bin of years

    # --- per experiment -----------------------------------------------------
    for exp_name in EXPERIMENT_NAMES:
        analyse_experiment(exp_name, BOX_SIZES)

    # --- outputs/ -----------------------------------------------------------
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    print(f"wrote {plot_compare_possTip(('Baseline', 'Widened'))}")

    # Monte Carlo: one list [alpha, mu, sigma2, tau, a] per parameter set, e.g.
    #   PARAM_SETS = [[a1, m1, s1, t1, aa1], [a2, m2, s2, t2, aa2], [a3, m3, s3, t3, aa3]]
    # Default example: the MLE, and the MLE with tau at each bound of its 95% interval.
    
    # careful, this MC experiments depend on the chosen liklihood!
    MC_EXPERIMENT = "Baseline"
    if experiment_is_complete(MC_EXPERIMENT):
        best = load_mle(MC_EXPERIMENT)["best_params"]
        profiles, grids = load_profiles(MC_EXPERIMENT)
        tau_lo, tau_up, _, _ = credibility_interval(grids["tau"], profiles["tau"])
        mle_set = [best[p] for p in params]
        PARAM_SETS = [
            mle_set,
            [3.06, 0.25, 0.55, 140.63, 0.86]
        ]
        print("wrote", *plot_monte_carlo(PARAM_SETS, num_sim=3, exp_name=MC_EXPERIMENT, seed=0))
        print(f"wrote {plot_compare_marginals(('Baseline', 'Widened'))}")
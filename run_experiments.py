"""
Possibility / necessity of AMOC tipping before time T, for several experiments.

The experiments only differ by their likelihood (see the LIKELIHOODS section).
For each experiment `<exp>`, a folder `<exp>/` is created containing:

    <exp>_mle.json                 MLE results (max log-lik, best params, derived params)
    <exp>_marginal_profiles.txt    every point of every marginal possibility function
    <exp>_poss_nec.txt             T, possibility, necessity, optimal theta and the
                                   marginal possibility of each optimal parameter value
    <exp>_marg.png                 marginal possibility profiles
    <exp>_possTip.png              possibility / necessity curves
    <exp>_evol_params.png          evolution of the optimal parameters

Each stage checks its file first: a complete file is reloaded instead of
recomputed, and an incomplete possibility/necessity file is resumed.
"""

import os

# Must be set BEFORE numpy is imported to take effect (also in the worker processes).
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import minimize
from tqdm import tqdm

# my code
from constants import *
#from metrics import *
from likelihoods import *
from utils import *
from plot_functions import *

# Fokker-Planck-based tipping-time distribution P(T_tip < T | theta)
from fp_tipping import proba_tip_before_fp


# =============================================================================
# LIKELIHOODS  (the only thing that changes between experiments)
# =============================================================================

from likelihoods import joint
from example import joint_flattened, joint_true

EXPERIMENTS = {
    "Baseline": joint,
    "Widened": joint_flattened,
    "FP_transition": joint_true,
}

# =============================================================================


params = ["alpha", "mu", "sigma2", "tau", "a"]
N_T_POINTS = 50


# =============================================================================
# Posterior (cached).
#
# posterior(theta) = exp(-likelihood(theta) - max_loglik_global) is the expensive
# ingredient shared by the possibility and necessity searches. Rounding theta
# before the cache lookup is safe because those searches use derivative-free
# Nelder-Mead. The experiment name is part of the key so that the caches of
# different likelihoods never mix.
# =============================================================================

@lru_cache(maxsize=20000)
def _cached_posterior(exp_name, theta_key, max_loglik_global):
    likelihood = EXPERIMENTS[exp_name]
    return float(np.exp(-likelihood(theta_key) - max_loglik_global))


def _posterior(theta, max_loglik_global, exp_name, round_dp=6):
    theta_key = tuple(round(float(v), round_dp) for v in theta)
    return _cached_posterior(exp_name, theta_key, max_loglik_global)


# =============================================================================
# Possibility
# =============================================================================

def possibility_theta_5d(theta, T, max_loglik_global, exp_name):
    """theta = [alpha, mu, sigma2, tau, a]"""
    posterior = _posterior(theta, max_loglik_global, exp_name)
    proba = proba_tip_before_fp(theta, T, t0)
    return posterior * proba


def possibility_tipping_before(T, values, max_loglik_global, exp_name):
    """Supremum over theta of posterior(theta) * P(T_tip < T | theta)."""
    if T <= 0:
        return {"possibility": 0.0, "best_theta_5d": values}

    def objective(theta):
        return -possibility_theta_5d(theta, T, max_loglik_global, exp_name)

    res = minimize(objective, x0=values, method="Nelder-Mead", bounds=b(params))

    return {"possibility": -res.fun, "best_theta_5d": res.x, "optimizer": res}


# =============================================================================
# Necessity
# =============================================================================

def possibility_contrary_theta_5d(theta, T, max_loglik_global, exp_name):
    posterior = _posterior(theta, max_loglik_global, exp_name)
    proba = proba_tip_before_fp(theta, T, t0)
    survival = 1 - proba
    return posterior * survival


def necessity_tipping_before(T, values, max_loglik_global, exp_name):
    """Necessity = 1 - Possibility(contrary event)."""

    def objective(theta):
        return -possibility_contrary_theta_5d(theta, T, max_loglik_global, exp_name)

    res = minimize(objective, x0=values, method="Nelder-Mead", bounds=b(params))

    max_poss_contrary = -res.fun
    nec = 1.0 - max_poss_contrary
    return nec


# =============================================================================
# Parallel workers (module level so that worker processes can import them)
# =============================================================================

def _possibility_worker(args):
    i, T, values, max_loglik_global, exp_name = args
    result = possibility_tipping_before(T, values, max_loglik_global, exp_name)
    return i, "possibility", result


def _necessity_worker(args):
    i, T, values, max_loglik_global, exp_name = args
    nec = necessity_tipping_before(T, values, max_loglik_global, exp_name)
    return i, "necessity", nec


# Detect cgroup quota (30 CPUs on Onyxia), fallback to affinity or CPU count
def get_allocated_cpus():
    # cgroup v2 check
    if os.path.exists("/sys/fs/cgroup/cpu.max"):
        try:
            with open("/sys/fs/cgroup/cpu.max", "r") as f:
                val = f.read().split()
                if val[0] != "max":
                    return int(int(val[0]) / int(val[1]))
        except Exception:
            pass
    # cgroup v1 check
    if os.path.exists("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
        try:
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f_q, \
                 open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f_p:
                quota = int(f_q.read().strip())
                period = int(f_p.read().strip())
                if quota > 0:
                    return int(quota / period)
        except Exception:
            pass
    # Fallback
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4)


# =============================================================================
# File helpers
# =============================================================================

def experiment_paths(exp_name):
    os.makedirs(exp_name, exist_ok=True)
    p = lambda suffix: os.path.join(exp_name, f"{exp_name}_{suffix}")
    return {
        "mle": p("mle.json"),
        "profiles": p("marginal_profiles.txt"),
        "poss_nec": p("poss_nec.txt"),
        "fig_marg": p("marg.png"),
        "fig_poss": p("possTip.png"),
        "fig_evol": p("evol_params.png"),
    }


def _atomic_write(path, text):
    """Write to a temporary file then rename: the file is either complete or absent."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def marginal_at(profiles, grids, p_name, value):
    return np.interp(value, grids[p_name], profiles[p_name])


# -----------------------------------------------------------------------------
# 1) MLE
# -----------------------------------------------------------------------------

def get_mle(exp_name, likelihood, path):
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        print(f"[{exp_name}] MLE loaded from {path}")
        return d["max_loglik_global"], d["best_params"], d["other_params"]

    print(f"[{exp_name}] Running MLE...")
    max_loglik_global, best_params, other_params = MLE(likelihood, params)
    d = {
        "max_loglik_global": float(max_loglik_global),
        "best_params": {k: float(v) for k, v in best_params.items()},
        "other_params": {k: float(v) for k, v in other_params.items()},
    }
    _atomic_write(path, json.dumps(d, indent=2))
    return d["max_loglik_global"], d["best_params"], d["other_params"]


# -----------------------------------------------------------------------------
# 2) Marginal possibility profiles
# -----------------------------------------------------------------------------

def _read_profiles(path):
    """Return the profiles dict if the file is complete, None otherwise."""
    if not os.path.exists(path):
        return None, None
    data = {p: ([], []) for p in params}
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) != 3 or cols[0] not in data:
                return None, None
            data[cols[0]][0].append(float(cols[1]))
            data[cols[0]][1].append(float(cols[2]))
    grids = {p: np.asarray(data[p][0]) for p in params}
    profiles = {p: np.asarray(data[p][1]) for p in params}
    return profiles, grids


def get_profiles(exp_name, likelihood, max_loglik_global, best_params, path):
    profiles, grids = _read_profiles(path)
    if profiles is not None:
        print(f"[{exp_name}] Marginal profiles loaded from {path}")
        return profiles, grids

    print(f"[{exp_name}] Computing marginal profiles...")
    profiles, grids = marg(likelihood, params, max_loglik_global, best_params)
    profiles = {p: np.asarray(profiles[p], dtype=float) for p in params}

    lines = ["# param\tvalue\tmarginal_possibility"]
    for p in params:
        for x, y in zip(grids[p], profiles[p]):
            lines.append(f"{p}\t{float(x):.8e}\t{float(y):.8e}")
    _atomic_write(path, "\n".join(lines) + "\n")
    return profiles, grids




# -----------------------------------------------------------------------------
# 3) Possibility / necessity (with checkpoint / resume)
# -----------------------------------------------------------------------------

N_COLS = 3 + 2 * len(params)  # T, poss, nec, theta (5), marginals (5)
POSS_NEC_HEADER = "# T\tpossibility\tnecessity\t" + "\t".join(params) + "\t" \
                  + "\t".join(f"marg_{p}" for p in params) + "\n"


def _format_row(T, poss, nec, theta, profiles, grids):
    margs = [marginal_at(profiles, grids, p, theta[k]) for k, p in enumerate(params)]
    return "\t".join(f"{float(v):.8e}" for v in [T, poss, nec, *theta, *margs]) + "\n"


def load_poss_nec(exp_name, path, T_values, poss_values, nec_values, opt):
    """Fill the arrays from the checkpoint file and return the index to resume from."""
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        with open(path, "w") as f:
            f.write(POSS_NEC_HEADER)
        return 0

    rows, corrupted = [], False
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) != N_COLS:   # e.g. a line truncated by a crash
                corrupted = True
                break
            rows.append(line)

    file_T = np.array([float(r.split()[0]) for r in rows])
    if len(rows) > len(T_values) or not np.allclose(file_T, T_values[:len(rows)], rtol=1e-6):
        backup = path + ".old"
        os.replace(path, backup)
        print(f"[{exp_name}] T grid of {path} does not match the current one: "
              f"moved to {backup}, starting from scratch.")
        with open(path, "w") as f:
            f.write(POSS_NEC_HEADER)
        return 0

    if corrupted:  # keep only the valid lines
        with open(path, "w") as f:
            f.write(POSS_NEC_HEADER)
            f.writelines(rows)

    for idx, r in enumerate(rows):
        cols = [float(c) for c in r.split()]
        poss_values[idx] = cols[1]
        nec_values[idx] = cols[2]
        for k, p in enumerate(params):
            opt[p][idx] = cols[3 + k]

    print(f"[{exp_name}] Checkpoint: {len(rows)}/{len(T_values)} points already calculated.")
    return len(rows)


def compute_poss_nec(exp_name, path, T_values, values, max_loglik_global, profiles, grids):
    N_pts = len(T_values)
    poss_values = np.zeros_like(T_values)
    nec_values = np.zeros_like(T_values)
    opt = {p: np.zeros_like(T_values) for p in params}

    start_idx = load_poss_nec(exp_name, path, T_values, poss_values, nec_values, opt)

    # ------------------------------------------------------------------------
    # All (T, branch) pairs are submitted as independent tasks; finished
    # results are buffered until BOTH branches of the next not-yet-written
    # index are ready, so the file is appended to in strict T order (needed
    # by the checkpoint/resume logic).
    # ------------------------------------------------------------------------

    pending = list(range(start_idx, N_pts))

    if pending:
        allocated_cpus = get_allocated_cpus()
        # Leave 1-2 cores for system, OS overhead, and the main dispatch process
        max_workers = max(1, allocated_cpus - 2)

        print(f"Detected {allocated_cpus} allocated cgroup CPUs.")
        print(f"[{exp_name}] Computing {len(pending)} T points x 2 (possibility, necessity) "
              f"on {max_workers} parallel worker processes...")

        results_poss, results_nec = {}, {}
        next_to_write = start_idx

        def _flush_ready(next_to_write):
            while next_to_write in results_poss and next_to_write in results_nec:
                i = next_to_write
                result = results_poss.pop(i)
                nec = results_nec.pop(i)

                poss_values[i] = result["possibility"]
                for k, p in enumerate(params):
                    opt[p][i] = result["best_theta_5d"][k]
                nec_values[i] = nec

                with open(path, "a") as f:
                    f.write(_format_row(T_values[i], poss_values[i], nec_values[i],
                                        [opt[p][i] for p in params], profiles, grids))
                    f.flush()
                next_to_write += 1
            return next_to_write

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = []
            for i in pending:
                T = T_values[i]
                futures.append(ex.submit(_possibility_worker, (i, T, values, max_loglik_global, exp_name)))
                futures.append(ex.submit(_necessity_worker, (i, T, values, max_loglik_global, exp_name)))

            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"{exp_name}: possibility & necessity (parallel)"):
                i, kind, payload = fut.result()
                (results_poss if kind == "possibility" else results_nec)[i] = payload
                next_to_write = _flush_ready(next_to_write)
    else:
        print(f"[{exp_name}] Possibility / necessity already complete.")

    return poss_values, nec_values, opt


# =============================================================================
# Plots
# =============================================================================

def plot_poss_nec(exp_name, T_values, poss_values, nec_values, t_c, path):
    plt.figure(figsize=(8, 5))
    plt.plot(T_values, poss_values, lw=2, color="blue", label=r"Credibility $\bar{P}(T_{tip} < t)$")
    plt.plot(T_values, nec_values, lw=2, color="orange", label=r"Necessity $N(T_{tip} < t)$")
    plt.axvline(x=t_c, color="red", linestyle="--", alpha=0.7, label=f"$t_c$ from MLE (~ {t_c:.1f})")
    plt.xlabel("Year")
    plt.ylabel("Credibility / Necessity")
    plt.title(f"{exp_name}: OMP of observing AMOC collapse before time t")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(path)
    plt.close()


def plot_evol_params(exp_name, T_values, opt, profiles, grids, t_c, path):
    cmap_byr = LinearSegmentedColormap.from_list("BlueYellowRed", ["blue", "yellow", "red"])
    labels = {"alpha": r"$\alpha$", "mu": r"$\mu$", "sigma2": r"$\sigma^2$", "tau": r"$\tau$", "a": r"$a$"}

    fig, axes = plt.subplots(nrows=len(params), ncols=1, figsize=(7, 7), sharex=True, layout="constrained")

    for ax, p_name in zip(axes, params):
        y_data = opt[p_name]
        marginal_values = marginal_at(profiles, grids, p_name, y_data)

        points = np.array([T_values, y_data]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        segment_colors = (marginal_values[:-1] + marginal_values[1:]) / 2

        lc = LineCollection(segments, cmap=cmap_byr, norm=plt.Normalize(0, 1))
        lc.set_array(segment_colors)
        lc.set_linewidth(2)

        line = ax.add_collection(lc)
        ax.autoscale()

        ax.axvline(x=t_c, color="red", linestyle="--", alpha=0.7)
        ax.set_ylabel(labels[p_name])
        ax.set_title(f"Evolution of {labels[p_name]}")
        ax.grid(alpha=0.3)

    fig.suptitle(exp_name)
    cbar = fig.colorbar(line, ax=axes, orientation="vertical")
    cbar.set_label("Marginal Possibility")

    plt.savefig(path)
    plt.close()


# =============================================================================
# One experiment
# =============================================================================

def run_experiment(exp_name):
    likelihood = EXPERIMENTS[exp_name]
    paths = experiment_paths(exp_name)

    # 1) MLE
    max_loglik_global, best_params, other_params = get_mle(exp_name, likelihood, paths["mle"])

    # 2) Marginal possibility profiles
    profiles, grids = get_profiles(exp_name, likelihood, max_loglik_global, best_params, paths["profiles"])
    plot_profiles(profiles, best_params, paths["fig_marg"], grids=grids)

    all_params = {**best_params, **other_params}
    t_c = all_params["t_c"]
    print(f"[{exp_name}] t_c value = {t_c}")
    values = [all_params[p] for p in params]

    # 3) Possibility / necessity
    time_for_predict = np.arange(2000, t_c + 20, 1 / 12)
    T_values = np.linspace(time_for_predict[0], time_for_predict[-1], num=N_T_POINTS)

    poss_values, nec_values, opt = compute_poss_nec(
        exp_name, paths["poss_nec"], T_values, values, max_loglik_global, profiles, grids
    )

    # 4) Plots
    plot_poss_nec(exp_name, T_values, poss_values, nec_values, t_c, paths["fig_poss"])
    plot_evol_params(exp_name, T_values, opt, profiles, grids, t_c, paths["fig_evol"])


if __name__ == "__main__":
    # Everything with side effects stays behind this guard: worker processes
    # re-import this file and must only pick up the definitions above.
    for exp_name in EXPERIMENTS:
        run_experiment(exp_name)
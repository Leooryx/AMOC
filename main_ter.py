import os
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
from tqdm import tqdm
from scipy.stats import norm
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.stats import wasserstein_distance
from scipy.optimize import differential_evolution

# my code
from constants import *
from metrics import *
from likelihoods import *
from utils import *
from plot_functions import *

# Fokker-Planck-based tipping-time distribution: replaces the quasi-static
# Eyring-Kramers hazard-rate approximation used further down.
from fp_tipping import proba_tip_before_fp

# (1) flattened-posterior objective, used everywhere `joint` used to be.
from example import joint_flattened


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


#np.random.seed(42)


params = ["alpha", "mu", "sigma2", "tau", "a"]


# ============================================================================
# Everything below that is a plain FUNCTION DEFINITION stays at module level
# (outside `if __name__ == "__main__":`), because the parallel workers below
# need to import this file to get these definitions -- see the note above
# the `if __name__ == "__main__":` block for why the rest of the script had
# to move under that guard.
# ============================================================================

# =============================================================================
# (2) Cheap, safe cache for the posterior term.
#
# posterior(theta) = exp(-joint_flattened(theta) - max_loglik_global) is the
# expensive, SHARED ingredient of both possibility_theta_5d and
# possibility_contrary_theta_5d (only the tipping-probability term differs
# between them). `joint_flattened` already carries its own exact-match cache
# (from example.py), which is safe but only catches bit-identical repeats.
# Here we can afford a little rounding on top of it: unlike the MLE/marg
# profiles (which use gradient-based L-BFGS-B and would break if a tiny
# finite-difference step got rounded away), the possibility/necessity search
# below uses derivative-free Nelder-Mead, so rounding theta to a few extra
# decimals before the cache lookup cannot silently zero out a gradient -- it
# only reuses a value for a theta that's already indistinguishable from one
# already evaluated. This is what actually gets reused: within one Nelder-Mead
# run (the simplex revisits nearby points during contraction/shrink steps)
# and, since ProcessPoolExecutor reuses worker processes across many
# submitted tasks, across different T's / possibility-vs-necessity tasks that
# happen to land on the same worker over the run.
# =============================================================================

@lru_cache(maxsize=20000)
def _cached_posterior(theta_key, max_loglik_global):
    return float(np.exp(-joint_flattened(theta_key) - max_loglik_global))


def _posterior(theta, max_loglik_global, round_dp=6):
    theta_key = tuple(round(float(v), round_dp) for v in theta)
    return _cached_posterior(theta_key, max_loglik_global)


# =============================================================================
# Possibility for one theta (Now uses full 5D parameter space)
# =============================================================================

def possibility_theta_5d(theta, T, max_loglik_global):
    """
    Evaluates possibility dynamically calculating t_c(theta) = tau.
    theta = [alpha, mu, sigma, tau, a]
    """
    alpha, mu, sigma, tau, a = theta

    posterior = _posterior(theta, max_loglik_global)

    proba = proba_tip_before_fp(theta, T, t0)

    return posterior * proba

# =============================================================================
# Global possibility (Optimizing over all 5 parameters)
# =============================================================================

def possibility_tipping_before(T, values, max_loglik_global):
    """
    Finds the supremum of the combined epistemic and aleatoric probability.
    """
    if T <= 0:
        return {"possibility": 0.0, "best_theta_5d": values}

    def objective(theta):
        # We minimize the negative possibility
        return -possibility_theta_5d(theta, T, max_loglik_global)

    res = minimize(objective, x0=values, method="Nelder-Mead", bounds=b(params))

    return {"possibility": -res.fun, "best_theta_5d": res.x, "optimizer": res}


# =============================================================================
# Same with necessity
# =============================================================================

def possibility_contrary_theta_5d(theta, T, max_loglik_global):

    alpha, mu, sigma, tau, a = theta

    # 1. Epistemic uncertainty
    posterior = _posterior(theta, max_loglik_global)

    # 2. Survival Probability (1 - CDF)
    proba = proba_tip_before_fp(theta, T, t0)

    survival = 1 - proba

    return posterior * survival



def necessity_tipping_before(T, values, max_loglik_global):
    """Calculates Necessity = 1 - Possibility(contrary event)."""

    def objective(theta):
        return -possibility_contrary_theta_5d(theta, T, max_loglik_global)

    res = minimize(objective, x0=values, method="Nelder-Mead", bounds=b(params))

    max_poss_contrary = -res.fun
    nec = 1.0 - max_poss_contrary
    return nec


# =============================================================================
# (2) Parallelization.
#
# For a given T, the possibility search and the necessity search are two
# independent Nelder-Mead optimizations (different objectives, no shared
# state) -- nothing stops them running at the same time. And different T's
# are independent of each other too. So instead of the original loop
# (T=0: possibility, then necessity; T=1: possibility, then necessity; ...),
# we submit ALL (T, branch) pairs -- 2 * N_pts independent tasks -- to one
# process pool and let it schedule them across however many workers the
# machine has, rather than fixing which worker does what.
#
# We deliberately do NOT dedicate a separate worker purely to "the
# posterior": posterior(theta) is not a standalone piece of work, it's a
# subroutine called from inside both possibility_theta_5d and
# possibility_contrary_theta_5d, typically hundreds of times per Nelder-Mead
# run. Routing every one of those calls through one fixed worker would
# serialize the single most-called piece of the whole computation onto one
# core -- the opposite of a speedup. The caching in _posterior above is the
# practical way to avoid recomputing it redundantly instead.
# =============================================================================

def _possibility_worker(args):
    i, T, values, max_loglik_global = args
    result = possibility_tipping_before(T, values, max_loglik_global)
    return i, "possibility", result


def _necessity_worker(args):
    i, T, values, max_loglik_global = args
    nec = necessity_tipping_before(T, values, max_loglik_global)
    return i, "necessity", nec


# ============================================================================
# (3) generated files renamed to carry "flattened_posterior_tipping_quantif"
# ============================================================================

RUN_TAG = "flattened_posterior_tipping_quantif"
SAVE_FILE = f"{RUN_TAG}.txt"


if __name__ == "__main__":
    # everything below has a side effect (runs the MLE optimization, launches
    # worker processes, writes files) so it belongs behind this guard: with
    # ProcessPoolExecutor, each worker re-imports this file, and without the
    # guard it would re-run this whole driver (and spawn its own pool) on
    # import instead of just picking up the function definitions above.

    function = joint_flattened
    max_loglik_global, best_params, other_params = MLE(function, params)

    #profiles = marg(function, params, max_loglik_global, best_params)

    #plot_profiles(profiles, best_params, f"{function.__name__}_marg.png")
    #computed_intervals = credibility_intervals(profiles=profiles, intervals=intervals, func_name=function.__name__, threshold=0.25, filename="estimation_metrics.txt")

    best_params.update(other_params)



    # Monte carlo estimation of noise-induced tippings distribution

    nsim = 5000
    Ttip = []

    alpha = best_params["alpha"]
    mu    = best_params["mu"]
    sigma2 = best_params["sigma2"]
    sigma = sigma2**0.5
    tau   = best_params["tau"]
    a     = best_params["a"]
    lambda0 = best_params["lambda0"] #need to concatenate dictionaries of estimated parameters!
    m = best_params["m"]
    t_c = best_params["t_c"]
    #print("t_c value = ", t_c)


    #short_time = from_t0[(2000 <= from_t0) & (from_t0 <= t_c + 5)]


    sim = False

    if sim:
        for i in tqdm(range(nsim)):
            X0 = m + np.sqrt(-lambda0 / a) + np.random.normal(0, 1)
            # we start not far from the stable position
            sim = X_traj(best_params, X0)
            Ttip.append(sim["FPT"])


        Ttip = np.array(Ttip)
        b_tipping = Ttip[Ttip == t_c]
        n_tipping = Ttip[Ttip != t_c]
        # For 2000 samples, 108 only have reached b-tipping (given the estimated value of t_c)
        print("Number of samples that have reach b-tipping: ", len(b_tipping))

        print("Median tipping year :", np.median(Ttip))
        print("2.5%, 16.5%, 50%, 83.5%, 97.5% quantiles for all tippings:")
        print(np.round(np.percentile(Ttip, [2.5, 16.5, 50, 83.5, 97.5]),1))
        print()
        print("Reminder, tc=", round(t_c,2))




    """# draw a line to show when the noise-induced tipping EK approx goes up again (sign that the approx is breaking down)
    short_time = np.array(from_t0[n//2:]) # for better visualization
    #short_time = short_time[short_time <= t_c]
    lamb_seq = [lambda_t_formula(t, lambda0, tau) for t in short_time]
    tau_noise_values = [mean_tau_noise_EK(lam, sigma, a, m) for lam in lamb_seq]
    # Assumption: the quasi-static EK approximation starts to break down when lamb_seq becomes increasing
    assumpt_break_down_idx = np.where(np.diff(tau_noise_values) > 0)[0][0] + 1 # careful about the syntax
    assumpt_break_down = short_time[assumpt_break_down_idx]

    time_for_tau_noise = short_time[short_time < t_c]
    plt.figure(figsize=(10, 6))
    plt.plot(time_for_tau_noise, tau_noise_values[:len(time_for_tau_noise)], 'b-', label='$\\tau_{n}$', linewidth=2)
    plt.title('Evolution of $ \\tau_{n}$ over time conditionally on the estimated $t_c$', fontsize=14)
    plt.axvline(t_c, color="red", linestyle=":", label="t_c")
    plt.axvline(assumpt_break_down, color="green", linestyle=":", label=f"potential break down of EF approx = {np.round(assumpt_break_down)}")
    plt.xlabel('Time from 1924', fontsize=12)
    plt.ylabel('$\\tau_{n}$', fontsize=12)
    plt.yscale('log') # bc tau_noise is HUGE at lambda0 unsurprinsingly
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=12)
    plt.savefig("images/mean_tau_noise_evolution.png")  # diagnostic, independent of joint vs joint_flattened -- left as-is
    plt.close()"""



    #x_minus = [m - np.sqrt(np.abs(Lambda)/a) for Lambda in lamb_seq]
    #x_plus = [m + np.sqrt(np.abs(Lambda)/a) for Lambda in lamb_seq]

    potential_barrier = []
    #for i in range(len(short_time)):
        #potential_barrier.append(U(x_minus[i], lamb_seq[i], a, m) - U(x_plus[i], lamb_seq[i], a, m))

    test_time = [t for t in range(1, 160)]
    values = [best_params["alpha"], best_params["mu"], best_params["sigma2"], best_params["tau"], best_params["a"]]



    full_time_seq = start_year + delta_t * np.arange(n)
    data_end_year = full_time_seq[-1]
    idx_t0 = np.searchsorted(full_time_seq, t0)
    start_to_t0 = full_time_seq[: idx_t0 + 1]
    future_time_seq = np.arange(data_end_year + delta_t, end_year + delta_t, delta_t)
    from_t0 = np.concatenate([full_time_seq[idx_t0:], future_time_seq])
    #short_time = np.array(from_t0[n//2:]) # for better visualization
    #from_t0 = short_time

    # =============================================================================
    # Tipping-time probability P(T_tip < T | theta) -- was the quasi-static
    # Eyring-Kramers hazard rate (tau_noise / hazard_rate), now solved directly
    # from the Fokker-Planck equation via proba_tip_before_fp() in fp_run.py
    # (imported at the top of this file).
    # =============================================================================

    # =============================================================================
    # Example & Possibility Curve Plotting
    # =============================================================================

    time_for_predict = np.arange(2000, t_c + 20, 1 / 12)
    T_values = np.linspace(time_for_predict[0], time_for_predict[-1], num=50) #we just select 50 points
    N_pts = len(T_values)

    poss_values = np.zeros_like(T_values)
    optimal_taus = np.zeros_like(T_values)
    optimal_alpha = np.zeros_like(T_values)
    optimal_mu = np.zeros_like(T_values)
    optimal_sigma2 = np.zeros_like(T_values)
    optimal_a = np.zeros_like(T_values)

    nec_values = np.zeros_like(T_values)

    start_idx = 0

    if os.path.exists(SAVE_FILE) and os.path.getsize(SAVE_FILE) > 0:
        # Read already computed lines (skipping comments or empty lines)
        with open(SAVE_FILE, "r") as f:
            lines = [line.strip().split() for line in f if line.strip() and not line.startswith("#")]

        start_idx = len(lines)

        # Repopulate memory from file
        for idx, cols in enumerate(lines):
            if idx >= N_pts:
                break
            # Format: T, poss, nec, alpha, mu, sigma2, tau, a
            poss_values[idx] = float(cols[1])
            nec_values[idx] = float(cols[2])
            optimal_alpha[idx] = float(cols[3])
            optimal_mu[idx] = float(cols[4])
            optimal_sigma2[idx] = float(cols[5])
            optimal_taus[idx] = float(cols[6])
            optimal_a[idx] = float(cols[7])

        print(f"Resuming from checkpoint: {start_idx}/{N_pts} points already calculated.")
    else:
        # Initialize new file with header
        with open(SAVE_FILE, "w") as f:
            f.write("# T\tpossibility\tnecessity\talpha\tmu\tsigma2\ttau\ta\n")


    # ------------------------------------------------------------------------
    # (2) parallel replacement of the old sequential
    #         for i in range(start_idx, N_pts):
    #             result = possibility_tipping_before(T, values)
    #             ...
    #             nec_values[i] = necessity_tipping_before(T, values)
    #     Both branches of every remaining T are submitted as independent
    #     tasks up front; a small buffer (`results_poss`/`results_nec`) holds
    #     finished results until BOTH branches of the next not-yet-written
    #     index are ready, so the file is still appended to in strict T order
    #     (needed for the checkpoint/resume logic above, which assumes line k
    #     of the file is T_values[k]) even though the work itself finishes out
    #     of order.
    # ------------------------------------------------------------------------

    pending = list(range(start_idx, N_pts))

    if pending:
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

        allocated_cpus = get_allocated_cpus()
        # Leave 1-2 cores for system, OS overhead, and the main dispatch process
        max_workers = max(1, allocated_cpus - 2)

        print(f"Detected {allocated_cpus} allocated cgroup CPUs.")
        print(f"Computing {len(pending)} T points x 2 (possibility, necessity) "
              f"on {max_workers} parallel worker processes...")

        results_poss, results_nec = {}, {}
        next_to_write = start_idx

        def _flush_ready(next_to_write):
            while next_to_write in results_poss and next_to_write in results_nec:
                i = next_to_write
                result = results_poss.pop(i)
                nec = results_nec.pop(i)

                poss_values[i] = result["possibility"]
                optimal_alpha[i] = result["best_theta_5d"][0]
                optimal_mu[i] = result["best_theta_5d"][1]
                optimal_sigma2[i] = result["best_theta_5d"][2]
                optimal_taus[i] = result["best_theta_5d"][3]
                optimal_a[i] = result["best_theta_5d"][4]
                nec_values[i] = nec

                with open(SAVE_FILE, "a") as f:
                    f.write(
                        f"{T_values[i]:.8e}\t{poss_values[i]:.8e}\t{nec_values[i]:.8e}\t"
                        f"{optimal_alpha[i]:.8e}\t{optimal_mu[i]:.8e}\t{optimal_sigma2[i]:.8e}\t"
                        f"{optimal_taus[i]:.8e}\t{optimal_a[i]:.8e}\n"
                    )
                    f.flush()
                next_to_write += 1
            return next_to_write

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = []
            for i in pending:
                T = T_values[i]
                futures.append(ex.submit(_possibility_worker, (i, T, values, max_loglik_global)))
                futures.append(ex.submit(_necessity_worker, (i, T, values, max_loglik_global)))

            for fut in tqdm(as_completed(futures), total=len(futures),
                             desc="Possibility & necessity (parallel)"):
                i, kind, payload = fut.result()
                (results_poss if kind == "possibility" else results_nec)[i] = payload
                next_to_write = _flush_ready(next_to_write)


    plt.figure(figsize=(8,5))
    plt.plot(T_values , poss_values, lw=2, color='blue', label="Credibility $\\bar{P}(T_{tip} < t)$")
    plt.plot(T_values ,nec_values,lw=2,color="orange",label=r"Necessity $N(T_{tip} < t)$",)
    plt.axvline(x=t_c, color='red', linestyle='--', alpha=0.7, label=f"Baseline $t_c$ (~ {t_c:.1f})")
    plt.xlabel("Year")
    plt.ylabel("Possibility / Necessity")
    plt.title("OMP of observing AMOC collapse before time t")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(f"images/{RUN_TAG}_possTip.png")
    plt.show()
    plt.close()


    years = T_values #t0_year + T_values


    evol = False
    if evol:
        cmap_byr = LinearSegmentedColormap.from_list("BlueYellowRed", ["blue", "yellow", "red"])

        list_opti = [
            {"data": optimal_alpha, "name": "alpha", "label": r"$\alpha$"},
            {"data": optimal_mu, "name": "mu", "label": r"$\mu$"},
            {"data": optimal_sigma2, "name": "sigma2", "label": r"$\sigma^2$"},
            {"data": optimal_taus, "name": "tau", "label": r"$\tau$"},
            {"data": optimal_a, "name": "a", "label": r"$a$"},
        ]

        fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(7, 7), sharex=True, layout="constrained")
        axes = axes.flatten()

        for i, param in enumerate(list_opti):
            ax = axes[i]

            y_data = param["data"]
            p_name = param["name"]

            marginal_values = np.interp(y_data, intervals[p_name], profiles[p_name])

            points = np.array([years, y_data]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            segment_colors = (marginal_values[:-1] + marginal_values[1:]) / 2

            lc = LineCollection(segments, cmap=cmap_byr, norm=plt.Normalize(0, 1))
            lc.set_array(segment_colors)
            lc.set_linewidth(2)

            line = ax.add_collection(lc)
            ax.autoscale()

            ax.axvline(x=t_c, color="red", linestyle="--", alpha=0.7, label=f"Baseline $t_c$ (~ {t_c:.1f})")
            ax.set_ylabel(param["label"])
            ax.set_title(f"Evolution of {param['label']}")
            ax.grid(alpha=0.3)

        cbar = fig.colorbar(line, ax=axes, orientation='vertical')
        cbar.set_label('Marginal Possibility')

        # save before show to avoid blank image!
        plt.savefig(f"images/{RUN_TAG}_evol_params.png")
        #plt.show()
        plt.close()
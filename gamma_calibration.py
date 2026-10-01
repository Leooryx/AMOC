"""
Calibrate gamma by simulating trajectories under theta_hat(gamma) and
comparing summary statistics to the real series.

For every gamma in GAMMA_GRID:
  1. find theta_hat(gamma) = argmax_theta flattened_loglik(theta, PARAMS, gamma=gamma)
  2. simulate NUM_SAMPLES trajectories under theta_hat(gamma), from start_year to T_END
  3. drop trajectories that tipped before T_END (and count how many)
  4. for every surviving trajectory compute:
       - beta0, beta1 of X(t) = beta0 + beta1 * t * 1{t > t0}
       - lag-1 and lag-12 cross-correlation with the real series
  5. print, for this gamma, the mean and 95% interval of each statistic

S_onestep / X_traj (utils.py) are used exactly as given, unmodified.

Two assumptions I had to make (flag if they are not what you meant):
  - X0 (the starting point of every simulated trajectory) = full_data[0],
    the first observed value, at start_year.
  - "lag-k autocorrelation between the sample and the true data" is read as
    the cross-correlation corr(sample[k:], true_data[:-k]) -- i.e. the
    sample at time t compared to the real data k steps earlier.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

from constants import *   # start_year, t0, delta_t, full_data, bounds, init_params, ...
from utils import *       # S_onestep, X_traj, b(...)
from example import flattened_loglik

PARAMS = ["alpha", "mu", "sigma2", "tau", "a"]
GAMMA_GRID = [0, 1000, 5000, 10000, 20000, 30000]


T_END = 2020.0      # horizon simulated for the calibration check
X0 = full_data[0]   # trajectories start from the first observed value, at start_year

NUM_SAMPLES = 1000     # <- start small to check everything works, then raise to 1000


# ---------------------------------------------------------------------------
# 1) theta -> the dict X_traj expects
# ---------------------------------------------------------------------------

def theta_to_traj_params(theta):
    alpha, mu, sigma2, tau, a = theta
    m = mu - alpha / (2.0 * a)
    lambda0 = -(alpha ** 2) / (4.0 * a)
    t_c = t0 + tau
    return {"alpha": alpha, "mu": mu, "sigma2": sigma2, "tau": tau, "a": a,
            "lambda0": lambda0, "m": m, "t_c": t_c}


# ---------------------------------------------------------------------------
# 2) MLE of theta for a fixed gamma (same bounds/start point used everywhere else)
# ---------------------------------------------------------------------------

def mle_for_gamma(gamma_value):
    def neg_loglik(theta):
        return -flattened_loglik(theta, PARAMS, gamma=gamma_value)

    res = minimize(neg_loglik, x0=init_params, method="Nelder-Mead",
                    bounds=b(PARAMS), options={"xatol": 1e-3, "fatol": 1e-3})
    return res.x


# ---------------------------------------------------------------------------
# 3) simulate one trajectory up to T_END and say whether it tipped before T_END
# ---------------------------------------------------------------------------

def simulate_until(theta, t_end=T_END):
    traj_params = theta_to_traj_params(theta)
    out = X_traj(traj_params, X0)          # unmodified; runs until tipping / t_c
    Xtraj = out["X"]

    time_grid = start_year + delta_t * np.arange(len(Xtraj))
    keep = time_grid <= t_end
    t_trunc = time_grid[keep]
    x_trunc = Xtraj[keep]

    lam = np.where(t_trunc < t0,
                   traj_params["lambda0"],
                   traj_params["lambda0"] * (1.0 - (t_trunc - t0) / traj_params["tau"]))
    barrier = traj_params["m"] - np.sqrt(np.maximum(-lam, 0.0) / traj_params["a"])
    tipped = bool(np.any(x_trunc <= barrier))

    return t_trunc, x_trunc, tipped


# ---------------------------------------------------------------------------
# 4) summary statistics
# ---------------------------------------------------------------------------

def regression_stats(t, x):
    indicator = (t > t0).astype(float)
    design = np.column_stack([np.ones_like(t), t * indicator])
    (beta0, beta1), *_ = np.linalg.lstsq(design, x, rcond=None)
    return beta0, beta1


def cross_corr(x, y, lag):
    n = min(len(x), len(y)) - lag
    return np.corrcoef(x[lag:lag + n], y[:n])[0, 1]


def sample_statistics(t, x):
    beta0, beta1 = regression_stats(t, x)
    n_overlap = min(len(x), len(full_data))
    acf1 = cross_corr(x[:n_overlap], full_data[:n_overlap], lag=1)
    acf12 = cross_corr(x[:n_overlap], full_data[:n_overlap], lag=12)
    return beta0, beta1, acf1, acf12


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    lo, hi = np.percentile(values, [2.5, 97.5])
    return mean, lo, hi


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    rows = []

    for gamma_value in tqdm(GAMMA_GRID):
        print(f"\n=== gamma = {gamma_value} ===")

        print("  fitting theta_hat(gamma)...")
        theta_hat = mle_for_gamma(gamma_value)
        print(f"  theta_hat = {dict(zip(PARAMS, np.round(theta_hat, 4)))}")

        if gamma_value == 0:
            print("  gamma=0 fully flattens the likelihood: theta_hat is not "
                  "identified, this run is only a reference point.")

        beta0s, beta1s, acf1s, acf12s = [], [], [], []
        n_tipped = 0

        for i in range(NUM_SAMPLES):
            t_traj, x_traj, tipped = simulate_until(theta_hat)
            if tipped:
                n_tipped += 1
                continue
            beta0, beta1, acf1, acf12 = sample_statistics(t_traj, x_traj)
            beta0s.append(beta0)
            beta1s.append(beta1)
            acf1s.append(acf1)
            acf12s.append(acf12)

            step = max(1, NUM_SAMPLES // 10)
            

        #print(f"  kept {NUM_SAMPLES - n_tipped}/{NUM_SAMPLES} samples and (removed {n_tipped} that tipped before t={T_END})")

        if NUM_SAMPLES - n_tipped == 0:
            print("  every sample tipped before T_END, skipping statistics for this gamma.")
            continue

        row = {"gamma": gamma_value, "n_tipped": n_tipped}
        for name, values in [("beta0", beta0s), ("beta1", beta1s),
                              ("acf_lag1", acf1s), ("acf_lag12", acf12s)]:
            mean, lo, hi = mean_ci(values)
            row[f"{name}_mean"] = mean
            row[f"{name}_CI95"] = (round(lo, 4), round(hi, 4))
        rows.append(row)

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print("\n\n=== summary table ===")
    print(table.to_string(index=False))

    table.to_csv("gamma_calibration_table.csv", index=False)
    print("\nsaved to gamma_calibration_table.csv")
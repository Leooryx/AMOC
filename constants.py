import pandas as pd
import numpy as np


df = pd.read_csv('AMOC.txt', sep=' ')
n = len(df)

delta_t = 1/12
start_year = 1870
t0 = 1924.0
end_year = 2100.0 

"""full_time_seq = start_year + delta_t * np.arange(n)
idx_t0 = np.searchsorted(full_time_seq, t0)
start_to_t0 = full_time_seq[:idx_t0 + 1]
future_time_seq = np.arange(t0 + delta_t, end_year + delta_t, delta_t)
from_t0 = np.concatenate([full_time_seq[idx_t0 + 1:], future_time_seq])"""



full_time_seq = start_year + delta_t * np.arange(n)
data_end_year = full_time_seq[-1]  
idx_t0 = np.searchsorted(full_time_seq, t0)
start_to_t0 = full_time_seq[: idx_t0 + 1]
future_time_seq = np.arange(data_end_year + delta_t, end_year + delta_t, delta_t)
from_t0 = np.concatenate([full_time_seq[idx_t0:], future_time_seq])
all_time = np.concatenate([full_time_seq, future_time_seq])


df_stat = df[df['time'] < t0].copy()
df_nonstat = df[df['time'] >= t0].copy()

full_data = df["AMOC2"].values
statio_data = df_stat["AMOC2"].values
nonstatio_data = df_nonstat["AMOC2"].values


bounds = {
    'alpha': (1.0, 5.0),
    'mu':    (0.0, 1.0),
    'sigma2': (0.01, 1.0),
    'tau':   (100.0, 220.0), #(100.0, 220.0) previously
    'a':     (0.1, 2.0), #(0.1, 2.0)
}

grid_points = 40

intervals = {
    'alpha': np.linspace(*bounds['alpha'], grid_points),
    'mu':    np.linspace(*bounds['mu'], grid_points),
    'sigma2': np.linspace(*bounds['sigma2'], grid_points), # TODO: choose between sigma and sigma2
    'tau':     np.linspace(*bounds['tau'], grid_points),
    'a':   np.linspace(*bounds['a'], grid_points),
}

# mid point of the bounds
init_params = [3.25, 0.25, 0.3, 110.0, 2]





# for transition solver

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

N_STAT, N_NONSTAT = 10, 25
denom_norm = 1.0 / (N_STAT + N_NONSTAT)   # for normalizing the D_B sum to a per-transition average

# FP-solver resolution for the true-model transitions entering D_B
FP_N, FP_STEPS = 100, 50 #TODO: i have divided by 2 the resolution for faster speed but should improve for better accuracy


def _pick(idx, k):
    if len(idx) <= k:
        return idx
    pos = np.unique(np.round(np.linspace(0, len(idx) - 1, k)).astype(int))
    return idx[pos]


# TODO: maybe can precompute this. 
def select_transitions(t_obs, n_stat=N_STAT, n_nonstat=N_NONSTAT):
    t_prev_all = np.asarray(t_obs[:-1], dtype=float)
    stat_idx = np.where(t_prev_all < t0)[0]
    nonstat_idx = np.where(t_prev_all >= t0)[0]
    return np.sort(np.concatenate([_pick(stat_idx, n_stat), _pick(nonstat_idx, n_nonstat)]))


IDX = select_transitions(T_OBS)
X_PREV, T_PREV, T_NEXT = X_OBS[IDX], T_OBS[IDX], T_OBS[IDX + 1]


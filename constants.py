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
    'tau':   (100.0, 220.0),
    'a':     (0.1, 2.0),
}

grid_points = 100

intervals = {
    'alpha': np.linspace(*bounds['alpha'], grid_points),
    'mu':    np.linspace(*bounds['mu'], grid_points),
    'sigma2': np.linspace(*bounds['sigma2'], grid_points), # TODO: choose between sigma and sigma2
    'tau':     np.linspace(*bounds['tau'], grid_points),
    'a':   np.linspace(*bounds['a'], grid_points),
}

# mid point of the bounds
init_params = [3.25, 0.25, 0.3, 110.0, 2]


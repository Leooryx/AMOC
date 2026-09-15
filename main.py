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


#np.random.seed(42)


params = ["alpha", "mu", "sigma2", "tau", "a"]


function = joint
max_loglik_global, best_params, other_params = MLE(function, params)
profiles = marg(function, params, max_loglik_global, best_params)
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
print("t_c value = ", t_c)


short_time = from_t0[(2000 <= from_t0) & (from_t0 <= t_c + 5)]



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




# draw a line to show when the noise-induced tipping EK approx goes up again (sign that the approx is breaking down)
short_time = np.array(from_t0[n//2:]) # for better visualization
short_time = short_time[::10]
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
plt.savefig("images/mean_tau_noise_evolution.png")
plt.close()



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
# Vectorized Eyring-Kramers waiting time & Hazard (Unchanged)
# =============================================================================

def tau_noise(Lambda, sigma, A, m):
    Lambda = np.asarray(Lambda)
    x_minus = m - np.sqrt(np.abs(Lambda) / A)
    x_plus  = m + np.sqrt(np.abs(Lambda) / A)

    barrier = U(x_minus, Lambda, A, m) - U(x_plus, Lambda, A, m)

    return (2 * np.pi * np.exp(2 * barrier / sigma**2) / np.sqrt(U_2nd(x_plus, A, m) * np.abs(U_2nd(x_minus, A, m))))

def hazard_rate(t, theta):
    alpha, mu, sigma, tau, a = theta
    m = mu - alpha / (2 * a)
    lambda0 = -(alpha**2) / (4 * a)
    lam = lambda0 * (1 - (t-t0) / tau) # careful about time scale 
    
    r = np.full_like(lam, np.inf, dtype=float)
    mask = lam < 0

    if np.any(mask):
        r[mask] = 1.0 / tau_noise(lam[mask], sigma, a, m)
    return r

# =============================================================================
# Possibility for one theta (Now uses full 5D parameter space)
# =============================================================================

def possibility_theta_5d(theta, T):
    """
    Evaluates possibility dynamically calculating t_c(theta) = tau.
    theta = [alpha, mu, sigma, tau, a]
    """
    alpha, mu, sigma, tau, a = theta

    loglik = -joint(theta)
    posterior = np.exp(loglik - max_loglik_global)


    if T >= tau+t0: #tau: # why not try with absolute time t_c
        proba = 1.0
    else:
        t_grid = np.linspace(t0, T, 500) # be sure that all the time intervals are the same. 
        r = hazard_rate(t_grid, theta)
        H = cumulative_trapezoid(r, t_grid, initial=0)
        proba = 1.0 - np.exp(-H[-1])

    return posterior * proba

# =============================================================================
# Global possibility (Optimizing over all 5 parameters)
# =============================================================================

def possibility_tipping_before(T, values):
    """
    Finds the supremum of the combined epistemic and aleatoric probability.
    """
    if T <= 0:
        return {"possibility": 0.0, "best_theta_5d": values}

    def objective(theta):
        # We minimize the negative possibility
        return -possibility_theta_5d(theta, T)

    res = minimize(objective,x0=values,method="Nelder-Mead",bounds=b(params)) # #L-BFGS-B gave weird results

    return {"possibility": -res.fun,"best_theta_5d": res.x,"optimizer": res}


# =============================================================================
# Same with necessity
# =============================================================================

def possibility_contrary_theta_5d(theta, T):
    
    alpha, mu, sigma, tau, a = theta

    # 1. Epistemic uncertainty
    loglik = -joint(theta)
    posterior = np.exp(loglik - max_loglik_global)

    # 2. Survival Probability (1 - CDF)    
    if T >= tau+t0: 
        proba = 1.0
    else:
        t_grid = np.linspace(t0, T, 500) # be sure that all the time intervals are the same. 
        r = hazard_rate(t_grid, theta)
        H = cumulative_trapezoid(r, t_grid, initial=0)
        proba = 1.0 - np.exp(-H[-1])
    
    survival = 1 - proba

    return posterior * survival



def necessity_tipping_before(T, values):
    """Calculates Necessity = 1 - Possibility(contrary event)."""
    
    def objective(theta):
        return -possibility_contrary_theta_5d(theta, T)
    
    res = minimize(objective,x0=values,method="Nelder-Mead",bounds=b(params)) #L-BFGS-B gave weird results

    max_poss_contrary = -res.fun
    nec = 1.0 - max_poss_contrary
    return nec




# =============================================================================
# Example & Possibility Curve Plotting
# =============================================================================



T_max = t_c + 10 
T_values = from_t0

poss_values = np.zeros_like(T_values)
optimal_taus = np.zeros_like(T_values) 
optimal_alpha = np.zeros_like(T_values)
optimal_mu = np.zeros_like(T_values)
optimal_sigma2 = np.zeros_like(T_values)
optimal_a = np.zeros_like(T_values)


nec_values = np.zeros_like(T_values)

for i, T in enumerate(tqdm(T_values)):
    # Existing possibility calculation
    result = possibility_tipping_before(T, values)
    poss_values[i] = result["possibility"]
    optimal_alpha[i] = result["best_theta_5d"][0]
    optimal_mu[i] = result["best_theta_5d"][1]
    optimal_sigma2[i] = result["best_theta_5d"][2]
    optimal_taus[i] = result["best_theta_5d"][3]
    optimal_a[i] = result["best_theta_5d"][4]

    # New necessity calculation
    nec_values[i] = necessity_tipping_before(T, values)




plt.figure(figsize=(8,5))
plt.plot(from_t0 , poss_values, lw=2, color='blue', label="Upper Possibility of tipping $\\bar{P}(T_{tip} < T)$")
plt.plot(from_t0 ,nec_values,lw=2,color="orange",label=r"Lower Necessity $N(T_{tip} < T)$",)
plt.axvline(x=t_c, color='red', linestyle='--', alpha=0.7, label=f"Baseline $t_c$ (~ {t_c:.1f})")
plt.xlabel("Year")
plt.ylabel("Possibility / Upper Probability")
plt.title("OMP of observing AMOC collapse before time t")
plt.grid(alpha=0.3)
plt.legend()
plt.savefig("images/possTip.png")
plt.show()
plt.close()


years = from_t0 #t0_year + T_values




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
plt.savefig("images/evol_params.png")
plt.show()
plt.close()



# 1. Define 5-year bin edges across your time range
"""year_start = t0_year + T_values[0]
year_end = t0_year + T_values[-1]

bin_edges = np.arange(year_start, year_end + 5, 5)
bin_centers = bin_edges[:-1] + 2.5  # Center points for plotting bar charts

# 2. Interpolate cumulative values to exact bin boundary years
poss_cum_edges = np.interp(bin_edges, t0_year + T_values, poss_values)
nec_cum_edges = np.interp(bin_edges, t0_year + T_values, nec_values)

# 3. Calculate interval possibility and necessity for each 5-year bin
bin_possibility = poss_cum_edges[1:]  # Max possibility in (x, x+5]
bin_necessity = np.maximum(0, nec_cum_edges[1:])

# 4. Plot side-by-side grouped bar chart (histogram view)
plt.figure(figsize=(10, 5))
bar_width = 2.0  # Width of bars in years

plt.bar(
    bin_centers - bar_width / 2,
    bin_possibility,
    width=bar_width,
    color="skyblue",
    edgecolor="navy",
    label=r"Interval Possibility $\bar{P}(x < T < x+5)$",
    alpha=0.5
)
plt.bar(
    bin_centers + bar_width / 2,
    bin_necessity,
    width=bar_width,
    color="orange",
    edgecolor="darkred",
    label=r"Interval Necessity $N(x < T < x+5)$",
)

plt.axvline(
    x=tc_baseline_year,
    color="red",
    linestyle="--",
    alpha=0.7,
    label=f"Baseline $t_c$ (~ {tc_baseline_year:.1f})",
)

plt.xlabel("5-Year Window Start Year")
plt.ylabel("Probability Measure")
plt.title("5-Year Interval Bounds for AMOC Tipping Time $P(x < T < x+5)$")
plt.grid(alpha=0.3, axis="y")
plt.legend()
plt.tight_layout()
plt.show()"""
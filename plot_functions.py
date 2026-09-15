import numpy as np
from constants import *
from scipy.optimize import minimize
from scipy.integrate import cumulative_trapezoid
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d


def plot_profiles(profiles, estimates, pic_name):
    
    keys = list(intervals.keys())
    n_params = len(keys)

    labels = ['alpha', 'mu', 'sigma2', 'tau', 'a']
    default_colors = ['#204878', '#3870b0', '#6098d8', '#cc4c02', '#ec7014', '#8c510a']
    colors = default_colors[:n_params]

    fig, axes = plt.subplots(1, n_params, figsize=(3.6 * n_params, 4), sharey=True)


    for idx, key in enumerate(keys):
        ax = axes[idx]
        param_label = labels[idx]

        ax.plot(intervals[key], profiles[key], color=colors[idx], linewidth=2.5, label=r'$\pi(\theta|X)$')

    
        if key in estimates:
            x = estimates[key]
            ax.axvline(estimates[key], color='black', linestyle=':', label='MLE Global')
            ax.text(x, 0.4, f"{estimates[key]:.3g}",rotation=90,ha='left',va='top',fontsize=9,color='black')

        ax.set_title(param_label, fontsize=14, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=':', alpha=0.5)


        if idx == 0:
            ax.set_ylabel('Possibility ', fontsize=12)
            ax.legend(loc='upper right')

    plt.suptitle("Marginalized possibilities", fontsize=10, fontweight='bold')
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"images/{pic_name}")


def plot_credibility_intervals(all_intervals, all_best_params):

    colors = {
        "OU_negloglik": "#204878",
        "pseudo_negloglik": "#3870b0",
        "joint": "#cc4c02",
        "product": "#ec7014",
    }

    params = list(next(iter(all_intervals.values())).keys())
    functions = list(all_intervals.keys())

    fig, axes = plt.subplots(
        len(params),
        1,
        figsize=(7, 1.3*len(params)),
        constrained_layout=True
    )

    if len(params) == 1:
        axes = [axes]

    for ax, param in zip(axes, params):

        y = np.arange(len(functions))[::-1]

        for i, func in enumerate(functions):

            lb, ub = all_intervals[func][param]
            mle = all_best_params[func][param]

            ax.hlines(
                y=y[i],
                xmin=lb,
                xmax=ub,
                color=colors.get(func, "steelblue"),
                linewidth=4
            )

            ax.scatter(
                mle,
                y[i],
                color="black",
                s=35,
                zorder=3
            )

        ax.set_yticks(y)
        ax.set_yticklabels(functions)

        ax.set_title(param, fontsize=12, fontweight="bold")

        ax.grid(axis="x", linestyle=":", alpha=0.4)
    plt.savefig("images/credibility_intervals.png")


"""
time_for_tau_noise = short_time[short_time < t_c]
plt.figure(figsize=(10, 6))
plt.plot(time_for_tau_noise, tau_noise_values[:len(time_for_tau_noise)], 'b-', label='$\\tau_{noise}$', linewidth=2)
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


plt.figure(figsize=(8,5))
sns.histplot(n_tipping, bins=80, kde=True, stat="density", label="Monte Carlo estimation", alpha=0.5) 
plt.axvline(t_c, color="red", linewidth=2, label=f"Estimated deterministic bifurcation t_c (about {nsim - len(n_tipping)} samples)", linestyle=":")
plt.axvline(assumpt_break_down, color="green", lw=2, linestyle=":", label=f"potential break down of EK law = {np.round(assumpt_break_down)}")
plt.plot(short_time, density(short_time), color="red",lw=1,label="Quasi-static Eyring-Kramers")
plt.xlabel("Tipping year")
plt.ylabel("Density")
plt.title(f"Conditional empirical (Monte Carlo) VS theoretical (EK law) densities of n-tippings ({nsim} simulations)")
plt.legend()
plt.grid(alpha=0.5)
plt.savefig("images/MC_tip.png")
plt.close()



plt.figure(figsize=(10, 3))

# Small vertical jitter for visualization only
y = np.random.uniform(0, 0.02, size=len(Ttip))
plt.scatter(Ttip, y,s=10,alpha=0.6,color="steelblue")
plt.axvline(t_c, color="red", linestyle=":", linewidth=1,label=r"$t_c$")
plt.xlim(2050, 2062)
plt.ylim(-0.05, 0.05)
plt.yticks([])
plt.xlabel("Tipping year")
plt.title("Individual tipping times near the bifurcation")
plt.grid(axis="x", alpha=0.4)
plt.legend()
plt.tight_layout()
#plt.show()
plt.close()





t = t_c
info = proba_tipping_threshold(t, best_params, work_density=True)

cdf = info["cdf"]
density = info["density"]
proba = cdf(t)
print(proba)



plt.figure(figsize=(8,5))
plt.axvline(t_c, color="red", linewidth=2, label="Deterministic bifurcation", linestyle=":")
plt.axvline(assumpt_break_down, color="green", lw=2, linestyle=":", label=f"potential break down of EK law = {np.round(assumpt_break_down)}")
plt.plot(short_time, cdf(short_time), color="blue",lw=2,label="Quasi-static Eyring-Kramers cdf")
#plt.plot(short_time, density(short_time), color='red', lw=2, label="Density")
plt.xlabel("Tipping year")
plt.ylabel("CDF")
plt.title("Conditional quasi-static EK approx")
plt.legend()
plt.grid(alpha=0.5)
plt.savefig("images/cdf_EK.png")
plt.show()
plt.close()


plt.figure(figsize=(8,5))
plt.plot(short_time, potential_barrier / sigma2)
plt.xlabel("Time")
plt.ylabel("potential barrier / sigma2 for EK approx")
plt.axvline(assumpt_break_down, color="green", lw=2, linestyle=":", label=f"potential break down of EK law = {np.round(assumpt_break_down)}")
plt.axvline(t_c, color="red", linewidth=2, label="Deterministic bifurcation", linestyle=":")
plt.grid(True)
plt.title("Evolution of the ratio potential barrier / sigma2 for EK approx")
plt.savefig("images/evo_pot_bar.png")
plt.close()
"""
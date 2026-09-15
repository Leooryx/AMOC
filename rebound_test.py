from utils import *
from constants import *
import matplotlib.pyplot as plt
import pandas as pd 


params = ["alpha", "mu", "sigma2", "tau", "a"]


function = joint
max_loglik_global, best_params, other_params = MLE(function, params)
#profiles = marg(function, params, max_loglik_global, best_params)
#plot_profiles(profiles, best_params, f"{function.__name__}_marg.png")
#computed_intervals = credibility_intervals(profiles=profiles, intervals=intervals, func_name=function.__name__, threshold=0.25, filename="estimation_metrics.txt")

best_params.update(other_params)


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

filename = "data_sim.pkl"

if not(os.path.exists(filename)):
    results = []
    for i in tqdm(range(nsim)):
        X0 = m + np.sqrt(-lambda0 / a) + np.random.normal(0, 1) 
        res = simulate_trajectory(best_params, X0, t_start=1870, T_max=2080, dt=delta_t)
        results.append(res)
    data_sim = pd.DataFrame(results)
    data_sim.to_pickle(filename)
else:
    data_sim = pd.read_pickle(filename)





pure_bifurcation = ((data_sim['bifurcation_tip'] == True) & (data_sim['rebound'] == False)).sum()
pure_noise       = ((data_sim['noise_tip_before_tc'] == True) & (data_sim['rebound'] == False)).sum()
rebound_to_bifurcation = ((data_sim['bifurcation_tip'] == True) & (data_sim['rebound'] == True)).sum()
rebound_to_noise       = ((data_sim['noise_tip_before_tc'] == True) & (data_sim['rebound'] == True)).sum()

sizes = [pure_bifurcation, pure_noise, rebound_to_bifurcation, rebound_to_noise]
labels = [
    'Pure bifurcation', 
    'Pure noise', 
    'Rebound to Bifurcation', 
    'Rebound to Noise'
]
colors = ['#ff9999', '#66b3ff', '#ffcc99', '#99ff99']
explode = (0.05, 0, 0, 0) 

# Filter out categories with 0 counts so the pie chart remains clean
sizes, labels, colors = zip(*[(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0])

plt.figure(figsize=(8, 5))
plt.pie(
    sizes, 
    labels=labels, 
    colors=colors, 
    autopct='%1.1f%%', 
    startangle=140, 
    explode=explode[:len(sizes)], # adjust explode array length dynamically
    shadow=False,
    wedgeprops={'edgecolor': 'black'}
)

plt.title("Proportion of tipping types", fontsize=14, pad=20)
plt.axis('equal') 
plt.savefig("images/circular_tip_phenom.png")
#plt.show()
plt.close()



import random

def plot_category_trajectories(category_name, data, params, t_start=1870, dt=0.01, num_samples=2):
    
    # 1. Filter the dataframe based on the requested category
    if category_name == "Pure_bifurcation":
        mask = (data['bifurcation_tip'] == True) & (data['rebound'] == False)
    elif category_name == "Pure_noise":
        mask = (data['noise_tip_before_tc'] == True) & (data['rebound'] == False)
    elif category_name == "Rebound_bifurcation":
        mask = (data['bifurcation_tip'] == True) & (data['rebound'] == True)
    elif category_name == "Rebound_noise":
        mask = (data['noise_tip_before_tc'] == True) & (data['rebound'] == True)
    else:
        print(f"Error: Category '{category_name}' not recognized.")
        return

    filtered_data = data[mask]
    
    if len(filtered_data) == 0:
        print(f"No trajectories found for category: {category_name}")
        return
        
    # Randomly select up to num_samples indices
    sample_size = min(num_samples, len(filtered_data))
    sampled_indices = random.sample(list(filtered_data.index), sample_size)
    
    # 2. Reconstruct the theoretical curves up to t_c
    a = params["a"]
    m = params["m"]
    lambda0 = params["lambda0"]
    tau = params["tau"]
    t_c = t0 + tau

    # Time array just up to t_c for the theoretical barriers
    short_time = np.arange(t_start, t_c, dt)
    
    # Calculate lambda(t) for the short_time array
    lam_t = np.where(short_time < t0, lambda0, lambda0 * (1 - (short_time - t0) / tau))
    
    # Calculate stable and unstable states
    x_minus = m - np.sqrt(np.maximum(0, -lam_t / a))
    x_plus  = m + np.sqrt(np.maximum(0, -lam_t / a))

    # 3. Initialize the plot
    plt.figure(figsize=(10, 5))
    
    # Plot the sampled trajectories
    for idx in sampled_indices:
        X_traj = filtered_data.loc[idx, 'X']
        # Reconstruct the time array for this specific trajectory (it might have been truncated)
        t_traj = t_start + np.arange(len(X_traj)) * dt
        plt.plot(t_traj, X_traj, linewidth=1)

    # 4. Add the requested reference lines
    plt.plot(short_time, x_minus, color="blue", linewidth=1.5, label="Unstable state", linestyle="--")
    plt.plot(short_time, x_plus, color="blue", linewidth=1.5, label="Stable barrier")
    plt.axvline(x=t_c, color="red", linestyle="--", label=f"Critical time ($t_c={round(t_c, 1)}$)")
    
    # Formatting
    plt.title(f"Sampled Trajectories: {category_name} ({sample_size} shown)", fontsize=14)
    plt.xlabel("Years")
    plt.ylabel("System state (X)")
    plt.xlim(t_start, 2080)
    plt.ylim(-3.5, m + np.sqrt(-lambda0/a) + 2) # Adjust y-limits based on your data scale
    
    # Prevent duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')
    
    plt.grid(True, alpha=0.3)
    plt.savefig(f"images/example_{category_name}.png")
    #plt.show()
    plt.close()

# Pure_noise
# Pure_bifurcation
# Rebound_noise
# Rebound_bifurcation

for category_input in ["Pure_noise", "Pure_bifurcation", "Rebound_noise", "Rebound_bifurcation"]:
    plot_category_trajectories(category_input, data_sim, best_params, t_start=1870, dt=delta_t)





"""Ttip = np.array(Ttip)
b_tipping = Ttip[Ttip == t_c]
n_tipping = Ttip[Ttip != t_c]
# For 2000 samples, 108 only have reached b-tipping (given the estimated value of t_c)
print("Number of samples that have reach b-tipping: ", len(b_tipping))

print("Median tipping year :", np.median(Ttip))
print("2.5%, 16.5%, 50%, 83.5%, 97.5% quantiles for all tippings:")
print(np.round(np.percentile(Ttip, [2.5, 16.5, 50, 83.5, 97.5]),1))
print()
print("Reminder, tc=", round(t_c,2))

short_time = all_time[all_time <= t_c]
lamb_seq = [lambda_t_formula(t, lambda0, tau) for t in short_time]
tau_noise_values = [mean_tau_noise_EK(lam, sigma, a, m) for lam in lamb_seq]
# Assumption: the quasi-static EK approximation starts to break down when lamb_seq becomes increasing
#assumpt_break_down_idx = np.where(np.diff(tau_noise_values) > 0)[0][0] + 1 # careful about the syntax
#assumpt_break_down = short_time[assumpt_break_down_idx]

x_minus = [m - np.sqrt(np.abs(Lambda)/a) for Lambda in lamb_seq]
x_plus = [m + np.sqrt(np.abs(Lambda)/a) for Lambda in lamb_seq]

"""

#we former simulation code
# Number of samples that have reach b-tipping:  248 (means around 5% of samples reach b-tip)
# Median tipping year : 2051.0833333333335
# 2.5%, 16.5%, 50%, 83.5%, 97.5% quantiles for all tippings:
# [2031.9 2042.8 2051.1 2057.8 2061.2]







####### PLOT











"""plt.figure(figsize=(9, 5))
for idx, traj in enumerate(sample_trajectories):
    # Plot the trajectory
    plt.plot(all_time[:len(traj)], traj, alpha=0.4, linewidth=1)
    
    # Added: Check if the sample tipped before t_c (noise-induced tipping)
    t_tip_sample = sample_tipping_times[idx]
    if t_tip_sample != t_c:
        tipped_plotted_count += 1
        
        # Center the circle exactly where the simulation ends
        t_end = all_time[len(traj) - 1]
        x_end = traj[-1]
        
        # Add a label only to the first circle so we don't duplicate the legend
        circle_label = "Tipping point" if tipped_plotted_count == 1 else None
        
        # Plot little empty circles
        plt.plot(t_end, x_end, marker='o', markersize=7, 
                 markerfacecolor='none', markeredgecolor='red', 
                 markeredgewidth=1.5, linestyle='None', label=circle_label)

plt.plot(short_time, x_minus, color="blue", linewidth=1.5, label="Unstable state", linestyle="--")
plt.plot(short_time, x_plus, color="blue", linewidth=1.5, label="Stable barrier")
plt.axvline(x=t_c, color="red", linestyle="--", label=f"Critical time ($t_c={round(t_c, 1)}$)")

# Added: Calculate and print the proportion on the graph
proportion = tipped_plotted_count / n_samples_to_plot
text_str = f"Samples tipped before $t_c$: {proportion:.0%}"
# Place the text box in the bottom left (adjust the 0.05, 0.05 coordinates if it blocks your data)
plt.text(0.03, 0.05, text_str, transform=plt.gca().transAxes, fontsize=11,verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

plt.xlabel("Time (years)", fontsize=11)
plt.ylabel("Realizations", fontsize=11)
plt.title("Simulations with changing $\\lambda$", fontsize=13, fontweight="bold")
plt.grid(True, linestyle=":", alpha=0.4)
plt.legend(loc="upper right")

plt.show()"""

"""plt.figure(figsize=(9, 5))

for idx, traj in enumerate(sample_trajectories):
    # Make the rebound trajectory stand out slightly if it's the last one we appended
    is_rebound_traj = len(sample_rebound_indices[idx]) > 0
    alpha_val = 0.9 if is_rebound_traj else 0.3
    color_val = "black" if is_rebound_traj else "gray"
    
    plt.plot(all_time[:len(traj)], traj, alpha=alpha_val, linewidth=1, color=color_val)
    
    # Plot Tipping Points (Red Circles)
    t_tip_sample = sample_tipping_times[idx]
    if t_tip_sample != t_c: # Re-enabled this check so we don't plot circles on survivors!
        tipped_plotted_count += 1
        t_end = all_time[len(traj) - 1]
        x_end = traj[-1]
        
        circle_label = "Definitive Tipping" if tipped_plotted_count == 1 else None
        plt.plot(t_end, x_end, marker='o', markersize=7, 
                 markerfacecolor='none', markeredgecolor='red', 
                 markeredgewidth=1.5, linestyle='None', label=circle_label)

    # Plot Rebounds (Green X's)
    traj_rebounds = sample_rebound_indices[idx]
    for reb_idx in traj_rebounds:
        rebound_plotted_count += 1
        t_reb = all_time[reb_idx]
        x_reb = traj[reb_idx]
        
        reb_label = "Rebound detected" if rebound_plotted_count == 1 else None
        plt.plot(t_reb, x_reb, marker='x', markersize=8, 
                 color='green', markeredgewidth=2, linestyle='None', label=reb_label)

plt.plot(short_time, x_minus, color="blue", linewidth=1.5, label="Unstable state", linestyle="--")
plt.plot(short_time, x_plus, color="blue", linewidth=1.5, label="Stable barrier")
plt.axvline(x=t_c, color="red", linestyle="--", label=f"Critical time ($t_c={round(t_c, 1)}$)")

plt.xlabel("Time (years)", fontsize=11)
plt.ylabel("Realizations", fontsize=11)
plt.title("Simulations: Catching a Rare Rebound Event", fontsize=13, fontweight="bold")
plt.grid(True, linestyle=":", alpha=0.4)
plt.legend(loc="upper right")

plt.show()"""
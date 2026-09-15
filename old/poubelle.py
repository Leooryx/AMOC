OU_max_loglik, OU_best_params, OU_other_params = MLE(OU_negloglik, params)
P_max_loglik, P_best_params, P_other_params = MLE(pseudo_negloglik, params)

def product(params):

    # 1. Compute individual log possibilities (max value is 0)
    log_poss_OU = -OU_negloglik(params) - OU_max_loglik
    log_poss_P = -pseudo_negloglik(params) - P_max_loglik
    
    # 2. Product Rule on log scale is addition: log(pi_1 * pi_2) = log(pi_1) + log(pi_2)
    log_product = log_poss_OU + log_poss_P
    
    # 3. Return the negative so it can be minimized by scipy.optimize
    return -log_product


all_best_params[function.__name__] = best_params

all_intervals[function.__name__] = credibility_intervals(
    profiles=profiles,
    intervals=intervals,
    func_name=function.__name__,
    threshold=0.25,
    filename="estimation_metrics.txt"
)

plot_credibility_intervals(all_intervals, all_best_params)


def comp_poss(params):
    log_poss_OU = -OU_negloglik(params) - OU_max_loglik
    log_poss_P = -pseudo_negloglik(params) - P_max_loglik
    log_product = log_poss_OU + log_poss_P
    return -log_product

comparison = minimize(comp_poss, init_params, method='Nelder-Mead', bounds=b(params))

min_neg_log_product = comparison.fun

# it should be close from 1! But it is not...
max_possibility_product = np.exp(-min_neg_log_product)

print(f"Optimized Parameters: {comparison.x}")
print(f"Maximized Product of Possibilities: {max_possibility_product}")
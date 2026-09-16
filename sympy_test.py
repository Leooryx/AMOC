import sympy as sp
import numpy as np
from tqdm import tqdm
from utils import X_traj
import matplotlib.pyplot as plt


# part 1: simplified form 


# 1. Define symbolic variables
# u: x_{t+h} - mu_t
# X: x_t - mu_t
# rho: autocorrelation rho_t
# Omega: conditional variance Omega_t
# B: A * h / 2
u, X, rho, Omega, B = sp.symbols('u X rho Omega B', real=True)

# 2. Express the quantities in centered coordinates
# q_theta centered mean: Delta = rho * X
Delta = rho * X

# s_theta modified mean: Delta_tilde = rho * X / (1 + B * X)
Delta_tilde = (rho * X) / (1 + B * X)

# Z_{t+h} centered form: u / (1 - B * u) - Delta_tilde
Z = (u / (1 - B * u)) - Delta_tilde

# Jacobian factor: |d/du (phi_{h/2})^{-1}(u)| = 1 / (1 - B * u)^2
# Inside sqrt(q * s), sqrt(Jacobian) is:
sqrt_J = 1 / (1 - B * u)

# 3. Base Gaussian density q_theta(u)
q = (1 / sp.sqrt(2 * sp.pi * Omega)) * sp.exp(-(u - Delta)**2 / (2 * Omega))

# 4. Perturbation factor: sqrt(s_theta / q_theta)
# sqrt(q * s) = q * sqrt(s / q)
# Exponent term: -1/(4 * Omega) * [ Z^2 - (u - Delta)^2 ]
exponent_diff = -(Z**2 - (u - Delta)**2) / (4 * Omega)
perturbation = sqrt_J * sp.exp(exponent_diff)

# 5. Compute the Taylor expansion around B = 0 up to O(B^3)
# n=3 gives terms: order 0, order 1, and order 2
taylor_perturbation = perturbation.series(B, 0, n=3).removeO()

# Expand and collect in powers of B
taylor_poly = sp.expand(taylor_perturbation)
taylor_poly = sp.collect(taylor_poly, B)

# 6. Full integrand expansion: q(u) * (1 + c1*B + c2*B^2)
c0 = taylor_poly.coeff(B, 0)
c1 = taylor_poly.coeff(B, 1)
c2 = taylor_poly.coeff(B, 2)

print("Order 0 (Base Gaussian coefficient):")
print(sp.simplify(c0))

print("\nOrder 1 coefficient of B:")
print(sp.factor(c1))

print("\nOrder 2 coefficient of B^2:")
print(sp.simplify(c2))

# 7. (Optional) Integrate against Gaussian q_theta to get the Hellinger Affinity
# Substituting u = Delta + sqrt(Omega) * z where z ~ N(0, 1)
z = sp.symbols('z', real=True)
c1_z = sp.simplify(c1.subs(u, Delta + sp.sqrt(Omega) * z))
c2_z = sp.simplify(c2.subs(u, Delta + sp.sqrt(Omega) * z))

# Gaussian moments: E[z]=0, E[z^2]=1, E[z^3]=0, E[z^4]=3
def gaussian_expectation(expr, var=z):
    expr = sp.expand(expr)
    expr = expr.subs({
        var**6: 15,
        var**5: 0,
        var**4: 3,
        var**3: 0,
        var**2: 1,
        var: 0
    })
    return sp.simplify(expr)

E_c1 = gaussian_expectation(c1_z)
E_c2 = gaussian_expectation(c2_z)

print("\n--- Integrated Coefficients (Hellinger Affinity Expansion) ---")
print(f"E[c1] (should be 0): {E_c1}")
print(f"E[c2]: {E_c2}")




sigma = 0.53
lambda_0 = -2.49
t0 = 1924
tau_r = 137.00
A = 0.89
m = -1.44

h = 1/12
B_const = A * h / 2

t = 2000
x = 1.4

def Lambda(t):
    return lambda_0 * (1 - (t - t0) / tau_r)

def mu(t):
    return m + np.sqrt(-Lambda(t)/A)

def alpha(t):
    return 2 * np.sqrt(-Lambda(t)*A)

def rho_fun(t):
    return np.exp(-alpha(t)*h)

def gamma2_fun(t):
    return sigma**2 / (2 * alpha(t))
 
 
def Omega_fun(t):
    rho_t = rho_fun(t)
    return gamma2_fun(t) * (1 - rho_t**2)

rho = rho_fun(t)
gamma2 = sigma**2 / (2*alpha(t))
Omega = gamma2 * (1 - rho**2)
X = x - mu(t)



# what can x be equal too?????

def bound(t,x):
    rho = rho_fun(t)
    gamma2 = sigma**2 / (2*alpha(t))
    Omega = gamma2 * (1 - rho**2)
    X = x - mu(t)
    return B_const**2*(Omega*(-Omega - 22*X**2*rho**2 - 6*X**2*rho) - X**4*rho**4 - 2*X**4*rho**3 - X**4*rho**2)/(8*Omega)
print(- bound(t, x)) 
# 0.19738608595382603
print("hellinger:", np.sqrt(- bound(t, x)))
# 0.44428153906484347, très proche de 0.45!

print("")
print("Second part")
print("")

# part 2: real case
def m_t(x):
    return mu(t) + rho * (x - mu(t))

def q_theta(t, x, y):
    return (1 / np.sqrt(2 * np.pi * Omega)) * np.exp(-(y - m_t(x))**2 / (2 * Omega))

def phi(t, x):
    return mu(t) + (x - mu(t)) / (1+B_const*(x - mu(t)))

def phi_inv(t, y):
    return mu(t) + (y - mu(t)) / (1 - B_const * (y - mu(t)))

def z(t, x, y):
    return phi_inv(t, y) - m_t(phi(t, x))

def s_theta(t, x, y):
    return (1 / np.sqrt(2 * np.pi * Omega)) * np.exp(- z(t,x,y)**2 / (2 * Omega)) * (1 / (1 - B_const * (y - mu(t)))**2)

def H(q,s):
    return np.sqrt( 1 - np.trapezoid(np.sqrt(q*s), x=np.linspace(-1000, 1000, 1000)))


#### from claude

mu_sym, rho_sym, Omega_sym, B_sym = sp.symbols('mu_sym rho_sym Omega_sym B_sym', real=True)
x_sym, y_sym = sp.symbols('x_sym y_sym', real=True)
 
 
def m_t_sym(xx):
    return mu_sym + rho_sym * (xx - mu_sym)
 
 
def phi_sym(xx):
    return mu_sym + (xx - mu_sym) / (1 + B_sym * (xx - mu_sym))
 
 
def phi_inv_sym(yy):
    return mu_sym + (yy - mu_sym) / (1 - B_sym * (yy - mu_sym))
 
 
def z_sym_fun(xx, yy):
    return phi_inv_sym(yy) - m_t_sym(phi_sym(xx))
 
 
q_theta_sym = (1 / sp.sqrt(2 * sp.pi * Omega_sym)) * sp.exp(-(y_sym - m_t_sym(x_sym))**2 / (2 * Omega_sym))
Jac_sym = 1 / (1 - B_sym * (y_sym - mu_sym))**2
s_theta_sym = (1 / sp.sqrt(2 * sp.pi * Omega_sym)) * sp.exp(-z_sym_fun(x_sym, y_sym)**2 / (2 * Omega_sym)) * Jac_sym
 
sqrt_J_sym = 1 / (1 - B_sym * (y_sym - mu_sym))
exponent_diff_sym = -(z_sym_fun(x_sym, y_sym)**2 - (y_sym - m_t_sym(x_sym))**2) / (4 * Omega_sym)
perturbation_sym = sqrt_J_sym * sp.exp(exponent_diff_sym)
 
taylor_perturbation_real = perturbation_sym.series(B_sym, 0, n=3).removeO()
taylor_poly_real = sp.expand(taylor_perturbation_real)
taylor_poly_real = sp.collect(taylor_poly_real, B_sym)
 
c0_real = taylor_poly_real.coeff(B_sym, 0)
c1_real = taylor_poly_real.coeff(B_sym, 1)
c2_real = taylor_poly_real.coeff(B_sym, 2)
 
print("\n=== Part 2: real (uncentered) derivation ===")
print("Order 0:", sp.simplify(c0_real))
print("Order 1:", sp.factor(c1_real))
print("Order 2:", sp.simplify(c2_real))
 
# translation-invariance check: c2_real should not actually depend on mu_sym
# once expressed through (x - mu); a nonzero residual flags a real bug.
X_diff_sym = sp.symbols('X_diff_sym', real=True)
c2_real_centered = sp.simplify(c2_real.subs(x_sym, X_diff_sym + mu_sym))
residual_mu_dep = sp.simplify(c2_real_centered.diff(mu_sym))
print("Residual mu-dependence of c2 (should be 0):", residual_mu_dep)
 
# Gaussian expectation over y ~ N(m_t(x), Omega), i.e. y = m_t(x) + sqrt(Omega)*z_std
z_std = sp.symbols('z_std', real=True)
y_sub = m_t_sym(x_sym) + sp.sqrt(Omega_sym) * z_std
 
c1_real_z = sp.simplify(c1_real.subs(y_sym, y_sub))
c2_real_z = sp.simplify(c2_real.subs(y_sym, y_sub))
 
E_c1_real = gaussian_expectation(c1_real_z, z_std)
E_c2_real = gaussian_expectation(c2_real_z, z_std)
print(f"E[c1] real form (should be 0): {E_c1_real}")
 
H2_approxpr = sp.simplify(-(B_sym**2 * E_c2_real))
H2_approxpr = sp.simplify(H2_approxpr.subs(x_sym, X_diff_sym + mu_sym))
print("H(q,s)^2 second-order approximation (real form):")
print(H2_approxpr)
 
H2_approx_func = sp.lambdify((B_sym, Omega_sym, X_diff_sym, rho_sym), H2_approxpr, 'numpy')
 
 
def bound_H2_sympy(t, x):
    """Second-order-in-B analytic approximation to H(q_theta, s_theta)^2,
    derived directly from the real (uncentered) q_theta/s_theta forms."""
    Omega_t = Omega_fun(t)
    rho_t = rho_fun(t)
    mu_t = mu(t)
    return float(H2_approx_func(B_const, Omega_t, x - mu_t, rho_t))
 
 
def bound_H_sympy(t, x):
    return float(np.sqrt(max(0.0, bound_H2_sympy(t, x))))
 
 
# cross-check against the Part-1 hardcoded formula (both should agree)
_Omega_test = Omega_fun(t)
_rho_test = rho_fun(t)
_X_test = x - mu(t)
print("\nCross-check at (t=2000, x=1.4):")
print("  Part-1 hardcoded formula :", bound(t, x))
print("  Part-2 real-form sympy   :", bound_H2_sympy(t, x))
 
 
# ============================================================
# PART 2c: fixed numeric integral for H(q, s)
# ============================================================
 
def H_numeric(t, x, n_std=10, n_points=2001, margin=1e-6):
    """Numerically evaluate H(q_theta, s_theta) by integrating sqrt(q*s) over y,
    on a grid centered at the local mean and scaled to sqrt(Omega_t), staying
    clear of the Mobius map's pole at y = mu(t) + 1/B."""
    Omega_t = Omega_fun(t)
    std = np.sqrt(Omega_t)
    center = m_t(x)
 
    pole = mu(t) + 1.0 / B_const
    half_width = n_std * std
    y_max = min(center + half_width, pole - margin)
    y_min = center - half_width
 
    y_grid = np.linspace(y_min, y_max, n_points)
    q_vals = q_theta(t, x, y_grid)
    s_vals = s_theta(t, x, y_grid)
    integrand = np.sqrt(np.clip(q_vals * s_vals, 0.0, None))
    integral = np.trapezoid(integrand, x=y_grid)
    H2 = max(0.0, 1.0 - integral)
    return float(np.sqrt(H2)), float(integral)
 
 
# ============================================================
# PART 3: trajectory-based comparison, t = 1924 ... 2020
# ============================================================
 
delta_t = h  # ASSUMPTION: matches utils.py's module-level delta_t -- verify!
# t0 is already 1924 above -- ASSUMPTION: matches utils.py's module-level t0
 
params = {
    "alpha": alpha(t0),     # unpacked but unused in the X_traj body you gave;
    "mu": mu(t0),           # kept only for interface compatibility
    "sigma2": sigma**2,
    "tau": tau_r,
    "a": A,
    "lambda0": lambda_0,
    "m": m,
    "t_c": 2061.18,           # ASSUMPTION: no censoring time was given; set
                              # beyond 2020 so it doesn't truncate our window.
                              # Replace with your fitted t_c.
}
 
X0 = m + np.sqrt(-lambda_0 / A)  # stable branch equilibrium, t < t0
 
traj = X_traj(params, X0)
Xtraj = traj["X"]
times = 1870 + delta_t * np.arange(len(Xtraj))
 
mask = (times >= 1924) & (times <= 2020)
t_values = times[mask]
x_values = Xtraj[mask]
 
bound_vals = np.array([bound_H_sympy(tt, xx) for tt, xx in zip(t_values, x_values)])
H_vals = np.empty_like(bound_vals)
integral_vals = np.empty_like(bound_vals)
for i, (tt, xx) in enumerate(tqdm(zip(t_values, x_values), total=len(t_values))):
    H_i, integ_i = H_numeric(tt, xx)
    H_vals[i] = H_i
    integral_vals[i] = integ_i
 
# --- printed comparison (one row per year) ---
print("\n=== Bound vs numeric integral, 1924-2020 ===")
print(f"{'year':>8} {'x_t':>10} {'H_bound':>14} {'H_numeric':>14} {'abs diff':>12}")
for yr in range(1924, 2021):
    idx = int(np.argmin(np.abs(t_values - yr)))
    print(f"{t_values[idx]:8.2f} {x_values[idx]:10.4f} {bound_vals[idx]:14.6e} "
          f"{H_vals[idx]:14.6e} {abs(bound_vals[idx] - H_vals[idx]):12.3e}")
 
print(f"\nMean |bound - numeric|: {np.mean(np.abs(bound_vals - H_vals)):.6e}")
print(f"Max  |bound - numeric|: {np.max(np.abs(bound_vals - H_vals)):.6e}")
print(f"First passage time reported by X_traj: {traj['FPT']:.2f}  "
      f"(deterministic_tip={traj['deterministic_tip']}, rebound={traj['rebound']})")
 
# --- visualization ---
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
 
axes[0].plot(t_values, x_values, color="black", lw=1)
axes[0].set_ylabel("x_t")
axes[0].set_title("Simulated trajectory (X_traj)")
 
axes[1].plot(t_values, bound_vals, label="Analytic bound (2nd-order Taylor, sympy)", lw=2)
axes[1].plot(t_values, H_vals, label="Numeric H(q,s) (trapezoid integral)", lw=2, ls="--")
axes[1].set_xlabel("year t")
axes[1].set_ylabel("Hellinger distance H(q,s)")
axes[1].set_title("Analytic bound vs numeric Hellinger distance")
axes[1].legend()
 
plt.tight_layout()
plt.savefig("hellinger_bound_vs_integral.png", dpi=150)
plt.show()


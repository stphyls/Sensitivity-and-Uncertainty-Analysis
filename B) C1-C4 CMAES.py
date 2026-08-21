import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score

# =============================================================================
# 1. LOAD DATA
# =============================================================================
np.random.seed(42)
data = pd.read_csv('Downloads/Python files/Kuching_C4.csv')
date_cols = ['Year', 'Month', 'Day']
if all(col in data.columns for col in date_cols):
    data = data.sort_values(date_cols).reset_index(drop=True)
X = data[['MaxTemp', 'MinTemp', 'MeanTemp', 'Radiation', 'WS', 'Humidity']]
y = data['ET']

# =============================================================================
# 2. THREE-WAY SPLIT  —  70% Train / 15% Validation / 15% Test
#    Sequential split (chronological order preserved) — consistent with the
#    GSA script, and appropriate for time-ordered ET data.
#    Validation set : used by CMA-ES to evaluate fitness (prevents data leakage)
#    Test set       : reserved for final independent evaluation ONLY
# =============================================================================
n         = len(X)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

X_train, y_train = X.iloc[:train_end],       y.iloc[:train_end]
X_val,   y_val   = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
X_test,  y_test  = X.iloc[val_end:],          y.iloc[val_end:]

print("=" * 55)
print("  SVR-CMAES  |  Labuan C4 Configuration")
print("=" * 55)
print(f"  Total samples : {len(X)}")
print(f"  Training      : {len(X_train):>5}  ({len(X_train)/len(X):.1%})")
print(f"  Validation    : {len(X_val):>5}  ({len(X_val)/len(X):.1%})")
print(f"  Test          : {len(X_test):>5}  ({len(X_test)/len(X):.1%})\n")

# =============================================================================
# 3. DATA NORMALIZATION
#    MinMaxScaler fitted ONLY on training data to prevent data leakage.
#    SVR with RBF kernel is sensitive to feature scale differences
#    (e.g. temperature ~20–40 vs wind speed ~0–5), so normalization is mandatory.
# =============================================================================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# Apply training scaler to validation and test — NO re-fitting
X_val_scaled  = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

y_val_arr  = y_val.values
y_test_arr = y_test.values

# =============================================================================
# 4. CMA-ES IMPLEMENTATION
#    fitness function evaluates on VALIDATION SET ONLY
#    The optimizer NEVER touches the test set.
# =============================================================================
def evaluate_on_validation(solution):
    """Train SVR on training data; compute fitness on VALIDATION set."""
    C_val     = np.clip(solution[0], 0.001, 1000.0)
    gamma_val = np.clip(solution[1], 0.0001, 10.0)
    model = SVR(C=C_val, gamma=gamma_val, kernel='rbf')
    model.fit(X_train_scaled, y_train_scaled)

    val_pred_scaled = model.predict(X_val_scaled)
    val_pred = scaler_y.inverse_transform(val_pred_scaled.reshape(-1, 1)).ravel()
    val_mse  = np.mean((y_val_arr - val_pred) ** 2)   # fitness = validation MSE
    return val_mse


def cmaes_optimize(func, x0, sigma0=0.5, max_iterations=50, pop_size=None):
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    Minimizes func starting from x0 with initial step size sigma0.

    Parameters
    ----------
    func           : callable returning scalar fitness (validation MSE)
    x0             : initial mean vector (1-D array)
    sigma0         : initial step size (standard deviation)
    max_iterations : number of generations
    pop_size       : offspring per generation (default: 4 + floor(3*ln(n)))

    Returns
    -------
    best_solution  : array of best hyperparameters found
    best_fitness   : validation MSE of best solution
    """
    n = len(x0)
    lam = pop_size if pop_size else int(4 + np.floor(3 * np.log(n)))
    mu  = lam // 2

    # Recombination weights
    weights    = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights   /= weights.sum()
    mu_eff     = 1.0 / np.sum(weights ** 2)

    # Step-size control
    c_sigma    = (mu_eff + 2) / (n + mu_eff + 5)
    d_sigma    = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
    chi_n      = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))

    # Covariance matrix adaptation
    c_c        = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    c_1        = 2 / ((n + 1.3)**2 + mu_eff)
    c_mu       = min(1 - c_1,
                     2 * (mu_eff - 2 + 1/mu_eff) / ((n + 2)**2 + mu_eff))

    # State variables
    mean   = np.array(x0, dtype=float)
    sigma  = sigma0
    p_c    = np.zeros(n)
    p_s    = np.zeros(n)
    C_mat  = np.eye(n)

    best_solution = mean.copy()
    best_fitness  = float('inf')

    for gen in range(max_iterations):
        # Sample offspring
        try:
            eigvals, eigvecs = np.linalg.eigh(C_mat)
            eigvals = np.maximum(eigvals, 1e-20)
            B       = eigvecs
            D       = np.diag(np.sqrt(eigvals))
            BD      = B @ D
        except np.linalg.LinAlgError:
            C_mat = np.eye(n)
            BD    = np.eye(n)

        z_k    = np.random.randn(lam, n)
        y_k    = z_k @ BD.T
        x_k    = mean + sigma * y_k

        # Enforce bounds: C in [0.001, 1000], gamma in [0.0001, 10]
        x_k[:, 0] = np.clip(x_k[:, 0], 0.001, 1000.0)
        x_k[:, 1] = np.clip(x_k[:, 1], 0.0001, 10.0)

        # Evaluate fitness (validation MSE)
        fitness  = np.array([func(x) for x in x_k])
        rank_idx = np.argsort(fitness)

        # Track best
        if fitness[rank_idx[0]] < best_fitness:
            best_fitness  = fitness[rank_idx[0]]
            best_solution = x_k[rank_idx[0]].copy()

        # Update mean
        x_best  = x_k[rank_idx[:mu]]
        y_best  = y_k[rank_idx[:mu]]
        mean    = np.sum(weights[:, None] * x_best, axis=0)

        # Step-size path (p_s)
        C_invsqrt = B @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-20))) @ B.T
        p_s       = ((1 - c_sigma) * p_s
                     + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
                     * C_invsqrt @ (np.sum(weights[:, None] * y_best, axis=0)))

        # Step-size update
        sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_s) / chi_n - 1))

        # Covariance path (p_c)
        h_sig = (np.linalg.norm(p_s) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))
                 < (1.4 + 2 / (n + 1)) * chi_n)
        p_c   = ((1 - c_c) * p_c
                 + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff)
                 * np.sum(weights[:, None] * y_best, axis=0))

        # Covariance matrix update
        delta_h = (1 - h_sig) * c_c * (2 - c_c)
        C_mat   = ((1 - c_1 - c_mu) * C_mat
                   + c_1 * (np.outer(p_c, p_c) + delta_h * C_mat)
                   + c_mu * sum(weights[i] * np.outer(y_best[i], y_best[i])
                                for i in range(mu)))

        print(f"  CMA-ES Generation {gen + 1:>3}/{max_iterations}"
              f"  |  Best Validation MSE: {best_fitness:.6f}"
              f"  |  sigma: {sigma:.5f}")

    return best_solution, best_fitness


# Initial guess: log-space centre of search bounds
x0 = np.array([10.0, 0.1])   # C=10, gamma=0.1
print("--- CMA-ES Hyperparameter Optimization (Validation Set) ---")
best_solution, best_val_mse = cmaes_optimize(
    evaluate_on_validation, x0, sigma0=1.0, max_iterations=50
)
best_C     = np.clip(best_solution[0], 0.001, 1000.0)
best_gamma = np.clip(best_solution[1], 0.0001, 10.0)

print(f"\n  Best C     : {best_C:.6f}")
print(f"  Best gamma : {best_gamma:.6f}")
print(f"  Best Validation MSE : {best_val_mse:.6f}\n")

# =============================================================================
# 5. FINAL MODEL  —  fit with best hyperparameters, then evaluate all three sets
# =============================================================================
final_model = SVR(C=best_C, gamma=best_gamma, kernel='rbf')
final_model.fit(X_train_scaled, y_train_scaled)


def predict_orig(model, X_scaled):
    pred_scaled = model.predict(X_scaled)
    return scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()


y_train_pred = predict_orig(final_model, X_train_scaled)
y_val_pred   = predict_orig(final_model, X_val_scaled)
y_test_pred  = predict_orig(final_model, X_test_scaled)

# =============================================================================
# 6. PERFORMANCE METRICS
# =============================================================================
def compute_metrics(y_obs, y_sim, label=""):
    mse      = np.mean((y_obs - y_sim) ** 2)
    mae      = mean_absolute_error(y_obs, y_sim)
    rmse     = np.sqrt(mse)
    r2       = r2_score(y_obs, y_sim)
    mean_obs = y_obs.mean()

    mse_baseline = np.mean((y_obs - mean_obs) ** 2)
    nsce         = 1 - mse / mse_baseline
    gpi          = 1 - (mse / (mse_baseline + mse))

    sum_abs = np.sum(np.abs(y_sim - y_obs))
    sum_ref = np.sum(np.abs(y_sim - mean_obs) + np.abs(y_obs - mean_obs))
    d1      = 1 - sum_abs / sum_ref

    r_corr   = np.corrcoef(y_obs, y_sim)[0, 1]
    alpha    = np.std(y_sim) / np.std(y_obs)
    beta_kge = np.mean(y_sim) / mean_obs
    kge      = 1 - np.sqrt((r_corr - 1)**2 + (alpha - 1)**2 + (beta_kge - 1)**2)

    std_obs = np.std(y_obs)
    std_sim = np.std(y_sim)

    if label:
        print(f"{'─'*52}")
        print(f"  Metrics — {label}")
        print(f"{'─'*52}")
        print(f"  MAE  : {mae:.4f}   |  RMSE : {rmse:.4f}")
        print(f"  MSE  : {mse:.4f}   |  R²   : {r2:.4f}")
        print(f"  NSCE : {nsce:.4f}  |  GPI  : {gpi:.4f}")
        print(f"  d1   : {d1:.4f}   |  KGE  : {kge:.4f}")
        print(f"  r    : {r_corr:.4f}  |  SD_obs : {std_obs:.4f}  |  SD_sim : {std_sim:.4f}\n")

    return dict(mse=mse, mae=mae, rmse=rmse, r2=r2,
                nsce=nsce, gpi=gpi, d1=d1, kge=kge,
                r=r_corr, std_obs=std_obs, std_sim=std_sim)


train_m = compute_metrics(y_train.values, y_train_pred, "Training Set")
val_m   = compute_metrics(y_val_arr,      y_val_pred,   "Validation Set")
test_m  = compute_metrics(y_test_arr,     y_test_pred,  "Test Set (Final)")

print("\n--- Train / Validation / Test Comparison (Overfitting Diagnostic) ---")
print(f"{'Metric':<8} {'Train':>10} {'Validation':>12} {'Test':>10}")
print("─" * 44)
for key in ['r2', 'rmse', 'mae', 'nsce', 'gpi', 'kge', 'd1']:
    print(f"{key.upper():<8} {train_m[key]:>10.4f} {val_m[key]:>12.4f} {test_m[key]:>10.4f}")

print(f"\n  Station    : Labuan")
print(f"  Config     : C4 (MaxTemp, MinTemp, MeanTemp)")
print(f"  Optimizer  : CMA-ES (Covariance Matrix Adaptation Evolution Strategy)")
print(f"  Kernel     : RBF  |  C={best_C:.4f}  |  gamma={best_gamma:.4f}")

# =============================================================================
# 7. DIAGNOSTIC PLOTS
# =============================================================================
sets   = [('Training',   y_train.values, y_train_pred, train_m),
          ('Validation', y_val_arr,      y_val_pred,   val_m),
          ('Test',       y_test_arr,     y_test_pred,  test_m)]
colors = ['steelblue', 'darkorange', 'seagreen']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Row 1: Observed vs Predicted
for col, (label, y_obs, y_sim, met) in enumerate(sets):
    ax = axes[0, col]
    ax.scatter(y_obs, y_sim, alpha=0.5, s=18, color=colors[col])
    lim = [min(y_obs.min(), y_sim.min()) - 0.1,
           max(y_obs.max(), y_sim.max()) + 0.1]
    ax.plot(lim, lim, 'r--', linewidth=1.5, label='1:1 line')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Observed ET₀ (mm/day)')
    ax.set_ylabel('Simulated ET₀ (mm/day)')
    ax.set_title(f'{label}  |  R²={met["r2"]:.4f}  RMSE={met["rmse"]:.4f}')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')

# Row 2: Residuals
for col, (label, y_obs, y_sim, met) in enumerate(sets):
    ax = axes[1, col]
    residuals = y_obs - y_sim
    ax.scatter(y_sim, residuals, alpha=0.5, s=18, color=colors[col])
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Simulated ET₀ (mm/day)')
    ax.set_ylabel('Residuals (Obs − Sim)')
    ax.set_title(f'{label} Residuals  |  MAE={met["mae"]:.4f}  GPI={met["gpi"]:.4f}')

plt.tight_layout()
plt.savefig('Downloads/Python files/SVR_CMAES_Labuan_C4_diagnostics.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved: SVR_CMAES_Labuan_C4_diagnostics.png")
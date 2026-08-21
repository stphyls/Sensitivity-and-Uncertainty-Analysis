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
data = pd.read_csv('Downloads/Python files/Miri_C4.csv')
date_cols = ['Year', 'Month', 'Day']
if all(col in data.columns for col in date_cols):
    data = data.sort_values(date_cols).reset_index(drop=True)
X = data[['MaxTemp', 'MinTemp', 'Radiation', 'WS', 'Humidity']]
y = data['ET']

# =============================================================================
# 2. THREE-WAY SPLIT  —  70% Train / 15% Validation / 15% Test
# =============================================================================
n         = len(X)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

X_train, y_train = X.iloc[:train_end],         y.iloc[:train_end]
X_val,   y_val   = X.iloc[train_end:val_end],   y.iloc[train_end:val_end]
X_test,  y_test  = X.iloc[val_end:],            y.iloc[val_end:]   # ← no trailing comma

# =============================================================================
# 3. DATA NORMALIZATION  —  fitted on training data ONLY
# =============================================================================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

X_val_scaled  = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

y_val_arr  = y_val.values
y_test_arr = y_test.values

# =============================================================================
# 4. GSA OPTIMIZATION  —  fitness uses VALIDATION MSE only
# =============================================================================
def evaluate_on_validation(solution):
    C_val, gamma_val, eps_val = solution
    model = SVR(
        C       = np.clip(C_val,     0.001,  100.0),
        gamma   = np.clip(gamma_val, 0.0001,  10.0),
        epsilon = np.clip(eps_val,   0.0001,   1.0),
        kernel  = 'rbf'
    )
    model.fit(X_train_scaled, y_train_scaled)
    val_pred_scaled = model.predict(X_val_scaled)
    val_pred = scaler_y.inverse_transform(val_pred_scaled.reshape(-1, 1)).ravel()
    return np.mean((y_val_arr - val_pred) ** 2)


def gsa_optimize(func, bounds, num_agents=10, max_iterations=50):
    dims       = len(bounds)
    agents     = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_agents, dims))
    velocities = np.zeros((num_agents, dims))
    best_solution = None
    best_fitness  = float('inf')

    for iteration in range(max_iterations):
        fitness = np.array([func(agent) for agent in agents])

        min_idx = np.argmin(fitness)
        if fitness[min_idx] < best_fitness:
            best_fitness  = fitness[min_idx]
            best_solution = agents[min_idx].copy()

        G      = 100 * (1 - iteration / max_iterations)
        masses = fitness.max() - fitness + 1e-10
        masses /= masses.sum()

        for i in range(num_agents):
            force = np.zeros(dims)
            for j in range(num_agents):
                if i != j:
                    dist      = np.linalg.norm(agents[i] - agents[j]) + 1e-10
                    direction = (agents[j] - agents[i]) / dist
                    force    += np.random.rand() * G * masses[j] * direction / dist
            velocities[i] = np.random.rand() * velocities[i] + force
            agents[i]     = np.clip(agents[i] + velocities[i], bounds[:, 0], bounds[:, 1])

        print(f"  GSA Iteration {iteration + 1:>3}/{max_iterations}"
              f"  |  Best Validation MSE: {best_fitness:.6f}")

    return best_solution, best_fitness   # no history


print("--- GSA Hyperparameter Optimization (Validation Set) ---")
bounds = np.array([
    [0.01, 1000.0],     # C
    [0.0001, 10.0],     # gamma
    [0.0001, 1.0]       # epsilon
]) 

best_solution, best_val_mse = gsa_optimize(evaluate_on_validation, bounds)
best_C     = np.clip(best_solution[0], 0.001, 100.0)
best_gamma = np.clip(best_solution[1], 0.001, 10.0)
best_eps   = np.clip(best_solution[2], 0.001, 1.0)

print(f"\n  Best C       : {best_C:.6f}")
print(f"  Best gamma   : {best_gamma:.6f}")
print(f"  Best epsilon : {best_eps:.6f}")
print(f"  Best Validation MSE : {best_val_mse:.6f}\n")

# =============================================================================
# 5. FINAL MODEL
# =============================================================================
final_model = SVR(C=best_C, gamma=best_gamma, epsilon=best_eps, kernel='rbf')
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

print(f"\n  Station    : Alor Setar")
print(f"  Config     : C4 (MaxTemp, MinTemp, MeanTemp, Radiation, WS, Humidity)")
print(f"  Optimizer  : GSA  |  Kernel: RBF")
print(f"  C={best_C:.4f}  gamma={best_gamma:.4f}  epsilon={best_eps:.4f}")

# =============================================================================
# 7. DIAGNOSTIC PLOTS
# =============================================================================
sets   = [('Training',   y_train.values, y_train_pred, train_m),
          ('Validation', y_val_arr,      y_val_pred,   val_m),
          ('Test',       y_test_arr,     y_test_pred,  test_m)]
colors = ['steelblue', 'darkorange', 'seagreen']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

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

for col, (label, y_obs, y_sim, met) in enumerate(sets):
    ax = axes[1, col]
    residuals = y_obs - y_sim
    ax.scatter(y_sim, residuals, alpha=0.5, s=18, color=colors[col])
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Simulated ET₀ (mm/day)')
    ax.set_ylabel('Residuals (Obs − Sim)')
    ax.set_title(f'{label} Residuals  |  MAE={met["mae"]:.4f}  GPI={met["gpi"]:.4f}')

plt.tight_layout()
plt.savefig('Downloads/Python files/SVR_GSA.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved: SVR_GSA.png")
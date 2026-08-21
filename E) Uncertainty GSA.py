import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.pipeline import FeatureUnion
import matplotlib.pyplot as plt
import time

# ======================================================================
# EDIT PER STATION: set the CSV path and the configuration identifier.
# ======================================================================
csv_path = 'Downloads/Python files/Labuan_C4.csv'
config = 'C4'

print(f"\n===== SVR-GSA UNCERTAINTY ANALYSIS  |  {config}  |  {csv_path} =====")
start_time = time.time()

feature_sets = {
    'C1': ['MaxTemp', 'MinTemp', 'MeanTemp'],
    'C2': ['MaxTemp', 'MinTemp', 'MeanTemp', 'Radiation'],
    'C3': ['MaxTemp', 'MinTemp', 'MeanTemp', 'Radiation', 'WS'],
    'C4': ['MaxTemp', 'MinTemp', 'MeanTemp', 'Radiation', 'WS', 'Humidity']
}
columns_to_impute = feature_sets[config] + ['ET']

# ----------------------------------------------------------------------
# Step 1: Load raw data
# ----------------------------------------------------------------------
print("\n1. RAW DATA INSPECTION")
raw_data = pd.read_csv(csv_path)
for junk in ['Unnamed: 10', 'Unnamed: 11']:
    if junk in raw_data.columns:
        raw_data = raw_data.drop(columns=[junk])
print("Raw data shape:", raw_data.shape)
print("Missing values:\n", raw_data.isna().sum())

# ----------------------------------------------------------------------
# Step 2: Filter years
# ----------------------------------------------------------------------
print("\n2. FILTERING (Year >= 2000)")
filtered_data = raw_data[raw_data['Year'] >= 2000].copy()
print("Filtered shape:", filtered_data.shape)

# ----------------------------------------------------------------------
# Step 3: Datetime index
# ----------------------------------------------------------------------
print("\n3. DATETIME CONVERSION")
filtered_data['Date'] = pd.to_datetime(filtered_data[['Year', 'Month', 'Day']])
data = filtered_data.groupby('Date').mean(numeric_only=True).reset_index()
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date').sort_index()
print("Date range:", data.index.min(), "to", data.index.max(), "| days:", len(data))

# ----------------------------------------------------------------------
# Step 4: Missing value handling (interpolation)
# ----------------------------------------------------------------------
print("\n4. MISSING VALUE HANDLING")
data[columns_to_impute] = data[columns_to_impute].interpolate(method='linear')
data[columns_to_impute] = data[columns_to_impute].ffill().bfill()
print("Remaining missing:\n", data[columns_to_impute].isna().sum())

# ----------------------------------------------------------------------
# Step 5: Monthly resampling
# ----------------------------------------------------------------------
print("\n5. MONTHLY RESAMPLING")
data_monthly = data.resample('ME').mean(numeric_only=True)
print("Monthly shape:", data_monthly.shape)

# ----------------------------------------------------------------------
# Step 6: Feature engineering
# ----------------------------------------------------------------------
print("\n6. FEATURE ENGINEERING")
data_monthly['Month'] = data_monthly.index.month
data_monthly['Month_sin'] = np.sin(2 * np.pi * data_monthly['Month'] / 12)
data_monthly['Month_cos'] = np.cos(2 * np.pi * data_monthly['Month'] / 12)
data_monthly['ET_lag1'] = data_monthly['ET'].shift(1)
data_monthly['ET_lag1'] = data_monthly['ET_lag1'].fillna(data_monthly['ET'].mean())

engineered_features = ['Month', 'ET_lag1', 'Month_sin', 'Month_cos']
X = data_monthly[feature_sets[config] + engineered_features]
y = data_monthly['ET']

# ======================================================================
# Steps 7-8: CHRONOLOGICAL 70/15/15 SPLIT + LEAKAGE-FREE IMPUTE/SCALE
#   - split first (time-ordered; ET_lag1 must not leak)
#   - imputer and scaler fitted on the TRAINING subset ONLY
# ======================================================================
print("\n7-8. CHRONOLOGICAL 70/15/15 SPLIT + TRAIN-ONLY IMPUTE/SCALE")

X = X.sort_index()
y = y.loc[X.index]

n = len(X)
i_train = int(0.70 * n)
i_val = int(0.85 * n)

X_train_raw, y_train_s = X.iloc[:i_train], y.iloc[:i_train]
X_val_raw,   y_val_s   = X.iloc[i_train:i_val], y.iloc[i_train:i_val]
X_test_raw,  y_test_s  = X.iloc[i_val:], y.iloc[i_val:]

print(f"Train {X_train_raw.shape} | Val {X_val_raw.shape} | Test {X_test_raw.shape}")
print(f"Train: {X_train_raw.index.min().date()} -> {X_train_raw.index.max().date()}")
print(f"Val  : {X_val_raw.index.min().date()} -> {X_val_raw.index.max().date()}")
print(f"Test : {X_test_raw.index.min().date()} -> {X_test_raw.index.max().date()}")

# Impute features (fit on train only)
transformer = FeatureUnion(transformer_list=[
    ('features', SimpleImputer(strategy='mean')),
    ('indicators', MissingIndicator())
])
X_train_imp = transformer.fit_transform(X_train_raw)
X_val_imp = transformer.transform(X_val_raw)
X_test_imp = transformer.transform(X_test_raw)

# Impute target (fit on train only)
y_imputer = SimpleImputer(strategy='mean')
y_train = y_imputer.fit_transform(y_train_s.values.reshape(-1, 1)).ravel()
y_val = y_imputer.transform(y_val_s.values.reshape(-1, 1)).ravel()
y_test = y_imputer.transform(y_test_s.values.reshape(-1, 1)).ravel()

# Scale features (fit on train only)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_imp)
X_val = scaler.transform(X_val_imp)
X_test = scaler.transform(X_test_imp)

# ======================================================================
# Step 9: GSA optimization  (fitness = VALIDATION MSE, test untouched)
# ======================================================================
print("\n9. GSA OPTIMIZATION (fitness on validation subset)")

def svm_model(solution):
    C, gamma = solution
    model = SVR(C=C, gamma=np.clip(gamma, 0.0001, 10.0), kernel='rbf')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)              # VALIDATION, not test
    mse = np.mean((y_val - y_pred) ** 2)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mean_obs = y_val.mean()
    d1 = 1 - (np.sum((y_val - y_pred) ** 2) /
              np.sum((np.abs(y_pred - mean_obs) + np.abs(y_val - mean_obs)) ** 2))
    r = np.corrcoef(y_val, y_pred)[0, 1]
    alpha = np.std(y_pred) / np.std(y_val)
    beta = np.mean(y_pred) / np.mean(y_val)
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return mse, mae, rmse, d1, kge

def gsa_optimize(func, bounds, num_agents=10, max_iterations=50):
    dimensions = len(bounds)
    agents = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_agents, dimensions))
    velocities = np.zeros((num_agents, dimensions))
    best_solution = None
    best_fitness = float('inf')
    best_metrics = None

    for iteration in range(max_iterations):
        fitness_and_metrics = [func(agent) for agent in agents]
        fitness = np.array([m[0] for m in fitness_and_metrics])  # MSE as fitness

        min_idx = np.argmin(fitness)
        if fitness[min_idx] < best_fitness:
            best_fitness = fitness[min_idx]
            best_solution = agents[min_idx].copy()
            best_metrics = fitness_and_metrics[min_idx]

        G = 100 * (1 - iteration / max_iterations)
        masses = fitness.max() - fitness + 1e-10
        masses /= masses.sum()

        for i in range(num_agents):
            force = np.zeros(dimensions)
            for j in range(num_agents):
                if i != j:
                    distance = np.linalg.norm(agents[i] - agents[j]) + 1e-10
                    direction = (agents[j] - agents[i]) / distance
                    force += np.random.rand() * G * masses[j] * direction / distance
            velocities[i] = np.random.rand() * velocities[i] + force
            agents[i] = np.clip(agents[i] + velocities[i], bounds[:, 0], bounds[:, 1])

        if (iteration + 1) % 10 == 0:
            print(f"  iter {iteration + 1}/{max_iterations} - best val MSE: {best_fitness:.4f}")

    return best_solution, best_fitness, best_metrics

bounds = np.array([[0.001, 100], [0.0001, 10]])
best_solution, best_val_mse, best_metrics = gsa_optimize(svm_model, bounds)
best_C, best_gamma = best_solution
print(f"Best (on validation) - C: {best_C:.4f}, gamma: {best_gamma:.4f}, val MSE: {best_val_mse:.4f}")

def make_svr():
    return SVR(C=best_C, gamma=np.clip(best_gamma, 0.0001, 10.0), kernel='rbf')

# ======================================================================
# Step 10: FINAL model refit on TRAIN+VAL, single evaluation on TEST
# ======================================================================
print("\n10. FINAL TEST EVALUATION")
X_fit = np.vstack([X_train, X_val])
y_fit = np.concatenate([y_train, y_val])
n_fit = len(X_fit)

final_model = make_svr().fit(X_fit, y_fit)
test_pred = final_model.predict(X_test)
print(f"Test R2: {r2_score(y_test, test_pred):.4f} | "
      f"RMSE: {np.sqrt(mean_squared_error(y_test, test_pred)):.4f} | "
      f"MAE: {mean_absolute_error(y_test, test_pred):.4f}")

# ======================================================================
# Step 11: BOOTSTRAP (parameter CI + predictive interval)
# ======================================================================
print("\n11. BOOTSTRAP UNCERTAINTY")
n_bootstraps = 1000
boot_point = np.zeros((len(X_test), n_bootstraps))
boot_pred = np.zeros((len(X_test), n_bootstraps))

for i in range(n_bootstraps):
    if i % 100 == 0:
        print(f"  bootstrap {i}/{n_bootstraps}")
    idx = np.random.choice(n_fit, n_fit, replace=True)
    m = make_svr().fit(X_fit[idx], y_fit[idx])
    p = m.predict(X_test)
    boot_point[:, i] = p
    resid = y_fit[idx] - m.predict(X_fit[idx])
    boot_pred[:, i] = p + np.random.choice(resid, size=len(X_test), replace=True)

mean_predictions = boot_point.mean(axis=1)
y_test_original = y_test

ci_lower = np.percentile(boot_point, 2.5, axis=1)
ci_upper = np.percentile(boot_point, 97.5, axis=1)
pi_lower = np.percentile(boot_pred, 2.5, axis=1)
pi_upper = np.percentile(boot_pred, 97.5, axis=1)

# ======================================================================
# Step 12: POINT METRICS (on the test subset)
# ======================================================================
print(f"\n12. FINAL POINT METRICS ({config}, SVR-GSA)")
y_pred = mean_predictions
mse = mean_squared_error(y_test_original, y_pred)
mae = mean_absolute_error(y_test_original, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_original, y_pred)

mean_observed = np.mean(y_test_original)
d1 = 1 - (np.sum((y_test_original - y_pred) ** 2) /
          np.sum((np.abs(y_pred - mean_observed) + np.abs(y_test_original - mean_observed)) ** 2))

eps = 1e-10
std_pred = np.std(y_pred) + eps
std_obs = np.std(y_test_original) + eps
r = np.corrcoef(y_test_original, y_pred)[0, 1]
alpha = std_pred / std_obs
beta = np.mean(y_pred) / np.mean(y_test_original)
kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
gpi = np.sqrt(max(0.01, r2) / max(0.01, kge))

print(f"MSE {mse:.4f} | MAE {mae:.4f} | RMSE {rmse:.4f} | R2 {r2:.4f} | "
      f"D1 {d1:.4f} | KGE {kge:.4f} | GPI {gpi:.4f}")

# ======================================================================
# Step 13: INTERVAL VALIDATION  (Reviewer Comment 5: PICP / MPIW / calibration)
# ======================================================================
print("\n13. INTERVAL VALIDATION (PICP / MPIW / calibration)")

def coverage(lower, upper):
    inside = (y_test_original >= lower) & (y_test_original <= upper)
    return inside.mean(), np.mean(upper - lower)

picp_ci, mpiw_ci = coverage(ci_lower, ci_upper)
picp_pi, mpiw_pi = coverage(pi_lower, pi_upper)
print(f"CI on the mean : PICP {picp_ci:.3f} | MPIW {mpiw_ci:.3f} mm/day")
print(f"95% Prediction : PICP {picp_pi:.3f} | MPIW {mpiw_pi:.3f} mm/day   <-- report this")
print(f"Relative MPIW (prediction): {mpiw_pi / np.mean(y_test_original) * 100:.2f}%")

print("\nCalibration (nominal -> empirical coverage, prediction intervals):")
print(f"{'Nominal':>8}{'Empirical':>11}{'MPIW':>9}")
for lv in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
    lo = np.percentile(boot_pred, 100 * (1 - lv) / 2, axis=1)
    hi = np.percentile(boot_pred, 100 * (1 + lv) / 2, axis=1)
    cov = ((y_test_original >= lo) & (y_test_original <= hi)).mean()
    print(f"{lv:8.2f}{cov:11.3f}{np.mean(hi - lo):9.3f}")

# ======================================================================
# Step 14: Plot (shade the PREDICTION interval, the one whose coverage you report)
# ======================================================================
print("\n14. PLOT")
plt.figure(figsize=(12, 6))
plt.plot(y_test_original, 'r-', label='Observed ET0')
plt.plot(mean_predictions, 'bo', label='Predicted ET0')
plt.fill_between(range(len(mean_predictions)), pi_lower, pi_upper,
                 color='blue', alpha=0.2, label='95% prediction interval')
plt.xlabel('Test sample index'); plt.ylabel('ET0 (mm/day)')
plt.title(f'SVR-GSA {config}: test predictions with 95% prediction interval')
plt.legend(); plt.grid(True); plt.ylim(0, 8)
plt.savefig(f'{config}_predictions_PI_SVR-GSA.png', dpi=200)
plt.close()

# ======================================================================
# Step 15: Export
# ======================================================================
pd.DataFrame({
    'Observed_ET': y_test_original, 'Predicted_ET': mean_predictions,
    'PI_Lower': pi_lower, 'PI_Upper': pi_upper,
    'CI_Lower': ci_lower, 'CI_Upper': ci_upper
}).to_csv(f'{config}_uncertainty_results_SVR-GSA.csv', index=False)

pd.DataFrame({
    'Configuration': [config], 'Model': ['SVR-GSA'],
    'MSE': [mse], 'MAE': [mae], 'RMSE': [rmse], 'R2': [r2],
    'D1': [d1], 'KGE': [kge], 'GPI': [gpi],
    'PICP_predinterval': [picp_pi], 'MPIW_predinterval': [mpiw_pi],
    'PICP_meanCI': [picp_ci], 'MPIW_meanCI': [mpiw_ci],
    'Rel_MPIW_pct': [mpiw_pi / np.mean(y_test_original) * 100]
}).to_csv(f'{config}_metrics_SVR-GSA.csv', index=False)

print(f"\nSaved results and metrics for {config} (SVR-GSA).")
print(f"Total time: {time.time() - start_time:.1f} s")
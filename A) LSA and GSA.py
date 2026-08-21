import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from SALib.sample import fast_sampler
from SALib.analyze import fast

# ── Configuration ────────────────────────────────────────────────────────────
CSV_PATH   = 'Downloads/Python files/PulauLangkawi_C4.csv'
STATION    = 'Pulau Langkawi'
SEED       = 42
N_SAMPLES  = 4000     # samples per factor
M_INTERF   = 4        # interference factor

np.random.seed(SEED)

# ── Load data ────────────────────────────────────────────────────────────────
data = pd.read_csv(CSV_PATH)

# MeanTemp deliberately excluded — derived from MaxTemp and MinTemp
X = data[['MaxTemp', 'MinTemp', 'Humidity', 'Radiation', 'WS']]
y = data['ET']


def calculate_eto(params):
    """FAO-56 Penman-Monteith reference evapotranspiration (mm/day)."""
    max_temp, min_temp, rel_humidity, solar_rad, wind_speed = params

    # Derived internally — not a free sensitivity factor
    mean_temp = (max_temp + min_temp) / 2.0

    # Constants
    albedo = 0.23
    sigma  = 4.903e-9     # Stefan-Boltzmann, MJ K-4 m-2 day-1
    G      = 0            # soil heat flux, negligible for daily step
    P      = 101.3        # atmospheric pressure, kPa (sea level assumption)
    gamma  = 0.000665 * P # psychrometric constant

    # Saturation vapour pressure
    es_max = 0.6108 * np.exp(17.27 * max_temp / (max_temp + 237.3))
    es_min = 0.6108 * np.exp(17.27 * min_temp / (min_temp + 237.3))
    es     = (es_max + es_min) / 2.0

    # Actual vapour pressure
    ea = es * rel_humidity / 100.0

    # Slope of the saturation vapour pressure curve
    delta = 4098 * (0.6108 * np.exp(17.27 * mean_temp / (mean_temp + 237.3))) \
            / (mean_temp + 237.3) ** 2

    # Net shortwave radiation
    Rns = (1 - albedo) * solar_rad

    # Clear-sky ratio, approximated with a fixed Ra of 35 MJ m-2 day-1
    Rs_Rso = np.clip(solar_rad / (0.75 * 35.0), 0, 1)

    # Net longwave radiation (FAO-56 Eq. 39)
    Rnl = sigma * ((max_temp + 273.16) ** 4 + (min_temp + 273.16) ** 4) / 2.0 \
          * (0.34 - 0.14 * np.sqrt(ea)) \
          * (1.35 * Rs_Rso - 0.35)

    Rn = Rns - Rnl

    numerator   = 0.408 * delta * (Rn - G) \
                  + gamma * (900 / (mean_temp + 273)) * wind_speed * (es - ea)
    denominator = delta + gamma * (1 + 0.34 * wind_speed)

    return max(0, numerator / denominator)


# ── Local sensitivity: normalized OAT coefficient ────────────────────────────
def oat_sensitivity_analysis(X, feature_names):
    results       = {}
    sensitivities = {}

    parameter_ranges = {
        'MaxTemp':   [-20, -10, 0, 10, 20],
        'MinTemp':   [-20, -10, 0, 10, 20],
        'Humidity':  [-15, -7.5, 0, 7.5, 15],
        'Radiation': [-20, -10, 0, 10, 20],
        'WS':        [-30, -15, 0, 15, 30],
    }

    # Baseline is identical for every factor — compute once, not per factor
    baseline_y = np.mean([calculate_eto(row.values) for _, row in X.iterrows()])

    for feature in feature_names:
        baseline_value = X[feature].mean()

        sensitivity = []
        changes     = []
        percentages = parameter_ranges[feature]

        for pct in percentages:
            if pct == 0:
                sensitivity.append(0.0)
                changes.append(0.0)
                continue

            change    = baseline_value * (pct / 100)
            new_value = baseline_value + change
            changes.append(change)

            X_modified          = X.copy()
            X_modified[feature] = new_value

            mean_modified_y = np.mean(
                [calculate_eto(row.values) for _, row in X_modified.iterrows()]
            )

            # Normalized sensitivity coefficient
            if baseline_y != 0 and baseline_value != 0:
                sc = ((mean_modified_y - baseline_y) / baseline_y) / (pct / 100)
            else:
                sc = 0.0
            sensitivity.append(sc)

        nonzero_sc             = [abs(s) for p, s in zip(percentages, sensitivity) if p != 0]
        avg_sensitivity        = np.mean(nonzero_sc) if nonzero_sc else 0.0
        sensitivities[feature] = avg_sensitivity
        results[feature]       = {
            'changes':             changes,
            'percentages':         percentages,
            'sensitivity':         sensitivity,
            'average_sensitivity': avg_sensitivity,
            'baseline':            baseline_value,
        }

    return results, sensitivities


print(f"\n=== LOCAL SENSITIVITY ANALYSIS (OAT) — {STATION} ===")
oat_results, oat_sensitivities = oat_sensitivity_analysis(X, X.columns)

sorted_oat = sorted(oat_sensitivities.items(), key=lambda kv: abs(kv[1]), reverse=True)
print("\nOAT Sensitivity Rankings (Normalized SC):")
for feature, sensitivity in sorted_oat:
    print(f"  {feature:12} : {sensitivity:.6f}")


# ── OAT plot — 5 factors in a 3x2 grid, sixth cell hidden ────────────────────
feature_names = list(oat_results.keys())
ncols, nrows  = 2, 3

axis_labels = {
    'MaxTemp':   'Maximum Temperature',
    'MinTemp':   'Minimum Temperature',
    'Humidity':  'Relative Humidity',
    'Radiation': 'Solar Radiation',
    'WS':        'Wind Speed',
}

fig, axes = plt.subplots(nrows, ncols, figsize=(13, 15))
fig.subplots_adjust(hspace=0.4, wspace=0.25)

for idx, feature in enumerate(feature_names):
    row, col = divmod(idx, ncols)
    ax       = axes[row][col]

    result = oat_results[feature]
    pcts   = result['percentages']
    scs    = result['sensitivity']

    pcts_nonzero = [p for p in pcts if p != 0]
    scs_nonzero  = [s for p, s in zip(pcts, scs) if p != 0]

    ax.plot(pcts_nonzero, scs_nonzero, '--', color='#5B9BD5', alpha=0.45,
            linewidth=1.2, zorder=1)
    ax.scatter(pcts_nonzero, scs_nonzero, color='#2E75B6', s=70,
               zorder=2, edgecolors='white', linewidths=0.6)

    ax.axhline(y=0, color='#888888', linewidth=0.8, linestyle='-',  alpha=0.5)
    ax.axvline(x=0, color='#888888', linewidth=0.8, linestyle='--', alpha=0.4)

    for x_val, y_val in zip(pcts_nonzero, scs_nonzero):
        ax.annotate(f'{y_val:.2f}',
                    xy=(x_val, y_val),
                    xytext=(0, 9), textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=9, color='#1a1a1a')

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune='both'))

    ax.set_xlabel('Percentage Change (%)', fontsize=9, labelpad=8)
    ax.set_ylabel('Sensitivity Coefficient (SC)', fontsize=9, labelpad=8)
    ax.set_title(axis_labels[feature], fontsize=9, pad=10)
    ax.tick_params(labelsize=8.5)
    ax.grid(True, alpha=0.25, linewidth=0.6)

    x_pad = abs(pcts_nonzero[-1]) * 0.3
    ax.set_xlim(min(pcts_nonzero) - x_pad, max(pcts_nonzero) + x_pad)

axes[2][1].set_visible(False)

plt.savefig('OAT_Sensitivity_Corrected.png', dpi=180, bbox_inches='tight')
plt.show()
print("\nOAT plot saved as: OAT_Sensitivity_Corrected.png")


# ── Global sensitivity: eFAST ────────────────────────────────────────────────
print(f"\n=== GLOBAL SENSITIVITY ANALYSIS (eFAST) — {STATION} ===")

problem = {
    'num_vars': len(X.columns),
    'names':    X.columns.tolist(),
    'bounds':   [[X[col].min(), X[col].max()] for col in X.columns],
}

# eFAST requires N > 4*M^2 = 64 for M = 4.
# N = 4000 selected on the basis of the convergence check below.
param_values = fast_sampler.sample(problem, N=N_SAMPLES, M=M_INTERF, seed=SEED)
Y            = np.array([calculate_eto(sample) for sample in param_values])
Si           = fast.analyze(problem, Y, M=M_INTERF, print_to_console=True, seed=SEED)

print(f"\n  Model evaluations: {len(param_values)} "
      f"({N_SAMPLES} per factor x {problem['num_vars']} factors)")

# ── Verify theoretical constraints ───────────────────────────────────────────
print("\n=== VERIFICATION OF eFAST RESULTS ===")
print(f"  Sum of first-order indices (should be <= 1.0): {sum(Si['S1']):.4f}")
all_valid = all(st >= s1 for s1, st in zip(Si['S1'], Si['ST']))
print(f"  All STi >= Si (should be True): {all_valid}")
for name, s1, st in zip(Si['names'], Si['S1'], Si['ST']):
    flag = '' if st >= s1 else '  <- VIOLATION'
    print(f"    {name:12}: S1={s1:.4f}, ST={st:.4f}, Interaction={st - s1:.4f}{flag}")


# ── eFAST plot — line plot ───────────────────────────────────────────────────
efast_names = Si['names']
S1_values   = np.array(Si['S1'])
ST_values   = np.array(Si['ST'])
x_pos       = np.arange(len(efast_names))

fig2, ax2 = plt.subplots(figsize=(10, 5))

ax2.plot(x_pos, S1_values, '--', color='#2E75B6', alpha=0.45,
         linewidth=1.2, zorder=1)
ax2.scatter(x_pos, S1_values, color='#2E75B6', s=80, zorder=2,
            edgecolors='white', linewidths=0.6, label='First-Order (S1)')

ax2.plot(x_pos, ST_values, '--', color='#ED7D31', alpha=0.45,
         linewidth=1.2, zorder=1)
ax2.scatter(x_pos, ST_values, color='#ED7D31', s=80, zorder=2,
            edgecolors='white', linewidths=0.6, label='Total-Order (ST)')

for x_val, s1, st in zip(x_pos, S1_values, ST_values):
    ax2.annotate(f'{s1:.2f}', xy=(x_val, s1),
                 xytext=(0, 9), textcoords='offset points',
                 ha='center', va='bottom', fontsize=8.5, color='#2E75B6')
    ax2.annotate(f'{st:.2f}', xy=(x_val, st),
                 xytext=(0, 9), textcoords='offset points',
                 ha='center', va='bottom', fontsize=8.5, color='#C55A11')

ax2.set_xticks(x_pos)
ax2.set_xticklabels(efast_names, fontsize=10)
ax2.set_ylabel('Sensitivity Index', fontsize=10)
ax2.set_title(f'eFAST Global Sensitivity Analysis — {STATION}\n'
              'First-Order and Total-Order Indices',
              fontsize=12, fontweight='bold')
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax2.set_ylim(0, max(ST_values.max(), 1.0) * 1.15)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.25, linewidth=0.6)
ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('eFAST_Sensitivity_Corrected.png', dpi=180, bbox_inches='tight')
plt.show()
print("\neFAST plot saved as: eFAST_Sensitivity_Corrected.png")


# ── Convergence check ────────────────────────────────────────────────────────
# Reuses the N = 4000 run already computed above; only 1000 and 2000 are new.
print("\n=== eFAST CONVERGENCE CHECK ===")
conv = {N_SAMPLES: {'S1': S1_values, 'ST': ST_values}}
print(f"  N={N_SAMPLES:>5}: sum(S1)={S1_values.sum():.4f}")

for N_test in [1000, 2000]:
    pv = fast_sampler.sample(problem, N=N_test, M=M_INTERF, seed=SEED)
    Yt = np.array([calculate_eto(s) for s in pv])
    St = fast.analyze(problem, Yt, M=M_INTERF, print_to_console=False, seed=SEED)
    conv[N_test] = {'S1': np.array(St['S1']), 'ST': np.array(St['ST'])}
    print(f"  N={N_test:>5}: sum(S1)={np.array(St['S1']).sum():.4f}")

d_S1 = np.abs(conv[N_SAMPLES]['S1'] - conv[2000]['S1'])
d_ST = np.abs(conv[N_SAMPLES]['ST'] - conv[2000]['ST'])

print(f"\n  Max |dS1| between N=2000 and N={N_SAMPLES}: {d_S1.max():.4f}")
print(f"  Max |dST| between N=2000 and N={N_SAMPLES}: {d_ST.max():.4f}")
print("\n  Per-factor deviations:")
for nm, a, b in zip(problem['names'], d_S1, d_ST):
    print(f"    {nm:12}: dS1={a:.4f}  dST={b:.4f}")

print(f"\n  Report the largest value observed across ALL 20 stations "
      f"in the manuscript, not this station alone.")
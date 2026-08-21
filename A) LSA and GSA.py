import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from SALib.sample import fast_sampler
from SALib.analyze import fast

# Load data
data = pd.read_csv('Downloads/Python files/PulauLangkawi_C4.csv')

# FIXED: Remove MeanTemp — it is derived from MaxTemp and MinTemp
X = data[['MaxTemp', 'MinTemp', 'Humidity', 'Radiation', 'WS']]
y = data['ET']

def calculate_eto(params):
    max_temp, min_temp, rel_humidity, solar_rad, wind_speed = params

    # Derived internally — not a free sensitivity input
    mean_temp = (max_temp + min_temp) / 2.0

    # Constants
    albedo = 0.23
    sigma  = 4.903e-9
    G      = 0
    P      = 101.3
    gamma  = 0.000665 * P

    # Saturation vapour pressure
    es_max = 0.6108 * np.exp(17.27 * max_temp / (max_temp + 237.3))
    es_min = 0.6108 * np.exp(17.27 * min_temp / (min_temp + 237.3))
    es     = (es_max + es_min) / 2.0

    # Actual vapour pressure
    ea = es * rel_humidity / 100.0

    # Slope of saturation vapour pressure curve
    delta = 4098 * (0.6108 * np.exp(17.27 * mean_temp / (mean_temp + 237.3))) \
            / (mean_temp + 237.3) ** 2

    # Net shortwave radiation
    Rns = (1 - albedo) * solar_rad

    # Clear-sky radiation ratio (FAO-56 simplified)
    Rs_Rso = np.clip(solar_rad / (0.75 * 35.0), 0, 1)

    # Net longwave radiation (FAO-56 Eq. 39)
    Rnl = sigma * ((max_temp + 273.16) ** 4 + (min_temp + 273.16) ** 4) / 2.0 \
          * (0.34 - 0.14 * np.sqrt(ea)) \
          * (1.35 * Rs_Rso - 0.35)

    # Net radiation
    Rn = Rns - Rnl

    # FAO-56 Penman-Monteith
    numerator   = 0.408 * delta * (Rn - G) + gamma * (900 / (mean_temp + 273)) \
                  * wind_speed * (es - ea)
    denominator = delta + gamma * (1 + 0.34 * wind_speed)
    eto         = numerator / denominator

    return max(0, eto)


# ── Normalized OAT sensitivity coefficient ────────────────────────────────────
def oat_sensitivity_analysis(X, y, feature_names):
    results       = {}
    sensitivities = {}

    parameter_ranges = {
        'MaxTemp':   [-20, -10, 0, 10, 20],
        'MinTemp':   [-20, -10, 0, 10, 20],
        'Humidity':  [-15, -7.5, 0, 7.5, 15],
        'Radiation': [-20, -10, 0, 10, 20],
        'WS':        [-30, -15, 0, 15, 30],
    }

    for feature in feature_names:
        baseline_value = X[feature].mean()

        # Baseline ETo across full dataset
        baseline_etos = [calculate_eto(row.values) for _, row in X.iterrows()]
        baseline_y    = np.mean(baseline_etos)

        sensitivity = []
        changes     = []
        percentages = parameter_ranges[feature]

        for pct in percentages:
            if pct == 0:
                sensitivity.append(0)
                changes.append(0)
                continue

            change    = baseline_value * (pct / 100)
            new_value = baseline_value + change
            changes.append(change)

            X_modified          = X.copy()
            X_modified[feature] = new_value

            modified_etos   = [calculate_eto(row.values) for _, row in X_modified.iterrows()]
            mean_modified_y = np.mean(modified_etos)

            # Normalized sensitivity coefficient
            if baseline_y != 0 and baseline_value != 0:
                sc = ((mean_modified_y - baseline_y) / baseline_y) / (pct / 100)
            else:
                sc = 0
            sensitivity.append(sc)

        avg_sensitivity        = np.mean([abs(s) for s in sensitivity if s != 0])
        sensitivities[feature] = avg_sensitivity
        results[feature]       = {
            'changes':             changes,
            'percentages':         percentages,
            'sensitivity':         sensitivity,
            'average_sensitivity': avg_sensitivity,
            'baseline':            baseline_value,
        }

    return results, sensitivities


# ── Run OAT ──────────────────────────────────────────────────────────────────
print("\n=== LOCAL SENSITIVITY ANALYSIS (OAT) ===")
oat_results, oat_sensitivities = oat_sensitivity_analysis(X, y, X.columns)

sorted_oat = sorted(oat_sensitivities.items(), key=lambda x: abs(x[1]), reverse=True)
print("\nOAT Sensitivity Rankings (Normalized SC):")
for feature, sensitivity in sorted_oat:
    print(f"{feature:20} : {sensitivity:.6f}")


# ── OAT Plot — 5 variables in a 3x2 grid (6th cell hidden) ──────────────────
feature_names = list(oat_results.keys())   # 5 variables
ncols = 2
nrows = 3                                  # ceil(5/2) = 3

axis_labels = {
    'MaxTemp':   'Maximum Temperature',
    'MinTemp':   'Minimum Temperature',
    'Humidity':  'Relative Humidity',
    'Radiation': 'Solar Radiation',
    'WS':        'Wind Speed',
}

# Create figure and axes grid
fig, axes = plt.subplots(nrows, ncols, figsize=(13, 15))
fig.subplots_adjust(hspace=0.4, wspace=0.25)

for idx, feature in enumerate(feature_names):
    row, col = divmod(idx, ncols)
    ax       = axes[row][col]

    result = oat_results[feature]
    pcts   = result['percentages']
    scs    = result['sensitivity']

    # Exclude zero-perturbation point from plot
    pcts_nonzero = [p for p in pcts if p != 0]
    scs_nonzero  = [s for p, s in zip(pcts, scs) if p != 0]

    # Dashed line + scatter markers
    ax.plot(pcts_nonzero, scs_nonzero, '--', color='#5B9BD5', alpha=0.45,
            linewidth=1.2, zorder=1)
    ax.scatter(pcts_nonzero, scs_nonzero, color='#2E75B6', s=70,
               zorder=2, edgecolors='white', linewidths=0.6)

    # Reference lines
    ax.axhline(y=0, color='#888888', linewidth=0.8, linestyle='-',  alpha=0.5)
    ax.axvline(x=0, color='#888888', linewidth=0.8, linestyle='--', alpha=0.4)

    # Annotate each point — 2 decimal places
    for x_val, y_val in zip(pcts_nonzero, scs_nonzero):
        ax.annotate(f'{y_val:.2f}',
                    xy=(x_val, y_val),
                    xytext=(0, 9), textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=9, color='#1a1a1a')

    # Y-axis: uniform 2 decimal places
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune='both'))

    ax.set_xlabel('Percentage Change (%)', fontsize=9, labelpad=8)
    ax.set_ylabel('Sensitivity Coefficient (SC)', fontsize=9, labelpad=8)
    ax.set_title(f'{axis_labels[feature]}', fontsize=9, pad=10)
    ax.tick_params(labelsize=8.5)
    ax.grid(True, alpha=0.25, linewidth=0.6)

    x_pad = abs(pcts_nonzero[-1]) * 0.3
    ax.set_xlim(min(pcts_nonzero) - x_pad, max(pcts_nonzero) + x_pad)

# Hide unused 6th subplot
axes[2][1].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('OAT_Sensitivity_Corrected.png', dpi=180, bbox_inches='tight')
plt.show()
print("OAT plot saved as: OAT_Sensitivity_Corrected.png")


# ── eFAST ─────────────────────────────────────────────────────────────────────
print("\n=== GLOBAL SENSITIVITY ANALYSIS (eFAST) ===")
problem = {
    'num_vars': len(X.columns),
    'names':    X.columns.tolist(),
    'bounds':   [[X[col].min(), X[col].max()] for col in X.columns],
}

# N >= 65*(2k-1) for k=5 -> N >= 585; use 4000 for reliability
param_values = fast_sampler.sample(problem, M=4, seed=42)
Y            = np.array([calculate_eto(sample) for sample in param_values])
Si = fast.analyze(problem, Y, M=4, print_to_console=True, seed=42)

# ── Verification ──────────────────────────────────────────────────────────────
print("\n=== VERIFICATION OF eFAST RESULTS ===")
print(f"Sum of First-Order Indices (should be <= 1.0): {sum(Si['S1']):.4f}")
all_valid = all(Si['ST'][i] >= Si['S1'][i] for i in range(len(Si['names'])))
print(f"All STi >= Si (should be True): {all_valid}")
for name, s1, st in zip(Si['names'], Si['S1'], Si['ST']):
    flag = '' if st >= s1 else '  <- VIOLATION'
    print(f"{name:15}: S1={s1:.4f}, ST={st:.4f}, Interaction={st - s1:.4f}{flag}")

# ── eFAST Plot — Line plot ────────────────────────────────────────────────────
efast_names = Si['names']
S1_values   = Si['S1']
ST_values   = Si['ST']

x_pos = np.arange(len(efast_names))

fig2, ax2 = plt.subplots(figsize=(10, 5))

# First-order line
ax2.plot(x_pos, S1_values, '--', color='#2E75B6', alpha=0.45,
         linewidth=1.2, zorder=1)
ax2.scatter(x_pos, S1_values, color='#2E75B6', s=80, zorder=2,
            edgecolors='white', linewidths=0.6, label='First-Order (S1)')

# Total-order line
ax2.plot(x_pos, ST_values, '--', color='#ED7D31', alpha=0.45,
         linewidth=1.2, zorder=1)
ax2.scatter(x_pos, ST_values, color='#ED7D31', s=80, zorder=2,
            edgecolors='white', linewidths=0.6, label='Total-Order (ST)')

# Annotate points — 2 decimal places, colour-matched to line
for x_val, s1, st in zip(x_pos, S1_values, ST_values):
    ax2.annotate(f'{s1:.2f}',
                 xy=(x_val, s1),
                 xytext=(0, 9), textcoords='offset points',
                 ha='center', va='bottom', fontsize=8.5, color='#2E75B6')
    ax2.annotate(f'{st:.2f}',
                 xy=(x_val, st),
                 xytext=(0, 9), textcoords='offset points',
                 ha='center', va='bottom', fontsize=8.5, color='#C55A11')

ax2.set_xticks(x_pos)
ax2.set_xticklabels(efast_names, fontsize=10)
ax2.set_ylabel('Sensitivity Index', fontsize=10)
ax2.set_title('eFAST Global Sensitivity Analysis\nFirst-Order and Total-Order Indices',
              fontsize=12, fontweight='bold')
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax2.set_ylim(0, max(max(ST_values), 1.0) * 1.15)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.25, linewidth=0.6)
ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('eFAST_Sensitivity_Corrected.png', dpi=180, bbox_inches='tight')
plt.show()
print("eFAST plot saved as: eFAST_Sensitivity_Corrected.png")
print("\n=== eFAST CONVERGENCE CHECK ===")
conv = {}
for N_test in [1000, 2000, 4000]:
    pv = fast_sampler.sample(problem, N=N_test, M=4, seed=42)
    Yt = np.array([calculate_eto(s) for s in pv])
    St = fast.analyze(problem, Yt, M=4, print_to_console=False, seed=42)
    conv[N_test] = {'S1': np.array(St['S1']), 'ST': np.array(St['ST'])}
    print(f"  N={N_test:>5}: sum(S1)={St['S1'].sum():.4f}")

d_S1 = np.abs(conv[4000]['S1'] - conv[2000]['S1'])
d_ST = np.abs(conv[4000]['ST'] - conv[2000]['ST'])
print(f"\n  Max |ΔS1| between N=2000 and N=4000: {d_S1.max():.4f}")
print(f"  Max |ΔST| between N=2000 and N=4000: {d_ST.max():.4f}")
for nm, a, b in zip(problem['names'], d_S1, d_ST):
    print(f"    {nm:12}: ΔS1={a:.4f}  ΔST={b:.4f}")
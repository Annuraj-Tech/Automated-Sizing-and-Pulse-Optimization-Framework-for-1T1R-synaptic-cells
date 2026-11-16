from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error


# ============================================================
# Configuration and data
# ============================================================

BASE = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd()
)

CSV_PATH = BASE / "1t1r_dataset.csv"

if not CSV_PATH.is_file():
    raise FileNotFoundError(
        f"Dataset not found: {CSV_PATH}\n"
        "Place the CSV beside this script or edit CSV_PATH."
    )

df = pd.read_csv(CSV_PATH)

required = [
    "v_bl",
    "t_pw_us",
    "w_init_nm",
    "w_final_nm",
    "delta_w_nm",
    "r_init_kohm",
    "r_final_kohm",
    "e_device_pj",
]

missing = [column for column in required if column not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

if not np.isfinite(df[required].to_numpy(dtype=float)).all():
    raise ValueError("Dataset contains missing or non-finite values.")

if (df["t_pw_us"] <= 0).any() or (df["e_device_pj"] <= 0).any():
    raise ValueError("Pulse width and energy must be positive.")

X = np.column_stack([
    df["v_bl"],
    np.log10(df["t_pw_us"])
])

Y = np.column_stack([
    df["r_final_kohm"],
    np.log10(df["e_device_pj"])
])

train_idx, test_idx = train_test_split(
    np.arange(len(df)),
    test_size=0.20,
    random_state=42
)

print(
    f"Training: {len(train_idx)} samples | "
    f"Testing: {len(test_idx)} samples"
)


# ============================================================
# Same joint MLP as the updated train_surrogate.py
# ============================================================

model = TransformedTargetRegressor(
    regressor=make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(64, 64),
            max_iter=5000,
            random_state=0
        )
    ),
    transformer=StandardScaler()
)

print("Training target-scaled joint MLP...", flush=True)
model.fit(X[train_idx], Y[train_idx])

predicted = model.predict(X[test_idx])

actual_r = Y[test_idx, 0]
predicted_r = predicted[:, 0]

actual_e = 10.0 ** Y[test_idx, 1]
predicted_e = 10.0 ** predicted[:, 1]

r_mae = mean_absolute_error(actual_r, predicted_r)
r_r2 = r2_score(actual_r, predicted_r)

e_mae = mean_absolute_error(actual_e, predicted_e)
e_r2 = r2_score(actual_e, predicted_e)
e_mape = np.mean(
    np.abs((actual_e - predicted_e) / actual_e)
) * 100.0

residual_r = actual_r - predicted_r


# ============================================================
# Approximate VTEAM reference
# ============================================================

# Uses the bitline amplitude as an approximation to device voltage.
# Ignores transistor voltage drop and pulse-edge contributions.
KOFF = 8e-13
VOFF = 0.0115
ALPHA_OFF = 3

voltage = df["v_bl"].to_numpy()

reference_rate_m_s = np.zeros(len(df))
above_threshold = voltage > VOFF

reference_rate_m_s[above_threshold] = (
    KOFF
    * (voltage[above_threshold] / VOFF - 1.0) ** ALPHA_OFF
)

reference_dw_nm = (
    reference_rate_m_s
    * df["t_pw_us"].to_numpy()
    * 1e-6
    * 1e9
)

simulated_dw_nm = df["delta_w_nm"].to_numpy()
resistance_change = (
    df["r_final_kohm"] - df["r_init_kohm"]
)


# ============================================================
# Six-panel figure
# ============================================================

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})

fig, axes = plt.subplots(
    2, 3,
    figsize=(17, 10),
    constrained_layout=True
)

fig.suptitle(
    "1T1R Simulation Dataset and ML Surrogate Analysis",
    fontsize=16
)


# (a) Approximate equation consistency check
ax = axes[0, 0]

valid = (reference_dw_nm > 0) & (simulated_dw_nm > 0)

if valid.any():
    ax.scatter(
        reference_dw_nm[valid],
        simulated_dw_nm[valid],
        s=12,
        alpha=0.55,
        color="tab:blue"
    )

    low = min(
        reference_dw_nm[valid].min(),
        simulated_dw_nm[valid].min()
    )
    high = max(
        reference_dw_nm[valid].max(),
        simulated_dw_nm[valid].max()
    )

    ax.plot([low, high], [low, high], "k--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")

ax.set_xlabel("Approximate VTEAM state change (nm)")
ax.set_ylabel("Simulated state change (nm)")
ax.set_title("(a) Equation consistency — approximate")
ax.text(
    0.04, 0.96,
    "Assumes device voltage ≈ bitline amplitude\n"
    "Ignores pulse edges and transistor drop",
    transform=ax.transAxes,
    va="top",
    fontsize=8,
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
)
ax.grid(alpha=0.2, which="both")


# (b) Observed programming response
ax = axes[0, 1]

scatter = ax.scatter(
    df["t_pw_us"],
    df["r_final_kohm"],
    c=df["v_bl"],
    cmap="viridis",
    s=15,
    alpha=0.7
)

ax.set_xscale("log")
ax.set_xlabel("Pulse width (µs)")
ax.set_ylabel("Final resistance (kΩ)")
ax.set_title("(b) Simulated programming response")
fig.colorbar(scatter, ax=ax, label="Bitline amplitude (V)")
ax.grid(alpha=0.2)


# (c) Programming energy versus resistance change
ax = axes[0, 2]

scatter = ax.scatter(
    resistance_change,
    df["e_device_pj"],
    c=df["v_bl"],
    cmap="plasma",
    s=15,
    alpha=0.7
)

ax.set_yscale("log")
ax.set_xlabel("Resistance increase (kΩ)")
ax.set_ylabel("Device energy (pJ)")
ax.set_title("(c) Energy cost of programming")
fig.colorbar(scatter, ax=ax, label="Bitline amplitude (V)")
ax.grid(alpha=0.2)


# (d) Dataset coverage
ax = axes[1, 0]

ax.hist(
    df["r_final_kohm"],
    bins=25,
    color="tab:purple",
    edgecolor="white",
    alpha=0.85
)

ax.set_xlabel("Final resistance (kΩ)")
ax.set_ylabel("Number of samples")
ax.set_title("(d) Resistance distribution")
ax.grid(axis="y", alpha=0.2)


# (e) Held-out resistance residuals
ax = axes[1, 1]

ax.scatter(
    actual_r,
    residual_r,
    s=20,
    alpha=0.65,
    color="tab:green"
)

ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Simulated final resistance (kΩ)")
ax.set_ylabel("Simulated − predicted (kΩ)")
ax.set_title("(e) Joint MLP: test resistance residuals")
ax.text(
    0.04, 0.96,
    f"MAE = {r_mae:.3f} kΩ\n"
    f"R² = {r_r2:.5f}\n"
    f"Test samples = {len(test_idx)}",
    transform=ax.transAxes,
    va="top",
    fontsize=9,
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
)
ax.grid(alpha=0.2)


# (f) Held-out energy predictions in physical units
ax = axes[1, 2]

ax.scatter(
    actual_e,
    predicted_e,
    s=20,
    alpha=0.65,
    color="tab:orange"
)

low = min(actual_e.min(), predicted_e.min())
high = max(actual_e.max(), predicted_e.max())

ax.plot([low, high], [low, high], "k--", linewidth=1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Simulated device energy (pJ)")
ax.set_ylabel("Predicted device energy (pJ)")
ax.set_title("(f) Joint MLP: test energy predictions")
ax.text(
    0.04, 0.96,
    f"MAE = {e_mae:.2f} pJ\n"
    f"MAPE = {e_mape:.2f}%\n"
    f"R² = {e_r2:.5f}",
    transform=ax.transAxes,
    va="top",
    fontsize=9,
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
)
ax.grid(alpha=0.2, which="both")


# ============================================================
# Save plots and supporting data
# ============================================================

fig.savefig(BASE / "report_figures.png", dpi=300)
fig.savefig(BASE / "report_figures.pdf")

reference_table = pd.DataFrame({
    "v_bl": df["v_bl"],
    "t_pw_us": df["t_pw_us"],
    "approx_reference_delta_w_nm": reference_dw_nm,
    "simulated_delta_w_nm": simulated_dw_nm,
})

reference_table.to_csv(
    BASE / "approx_vteam_comparison.csv",
    index=False
)

test_table = pd.DataFrame({
    "dataset_row_index": test_idx,
    "actual_r_kohm": actual_r,
    "predicted_r_kohm": predicted_r,
    "residual_r_kohm": residual_r,
    "actual_energy_pj": actual_e,
    "predicted_energy_pj": predicted_e,
})

test_table.to_csv(
    BASE / "extra_plots_test_predictions.csv",
    index=False
)

print(f"\nResistance MAE: {r_mae:.4f} kΩ")
print(f"Resistance R²: {r_r2:.6f}")
print(f"Energy MAE: {e_mae:.3f} pJ")
print(f"Energy MAPE: {e_mape:.3f}%")
print(f"\nFigures and supporting CSV files saved in: {BASE}")

plt.show()
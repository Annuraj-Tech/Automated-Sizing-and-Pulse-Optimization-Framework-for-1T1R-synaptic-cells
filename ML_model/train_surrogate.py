from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, WhiteKernel, ConstantKernel as C
)
from sklearn.metrics import r2_score, mean_absolute_error


# PC: use the script's folder. Colab: use the current folder.
BASE = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd()
)

df = pd.read_csv(BASE / "1t1r_dataset.csv")

# Inputs: voltage and log10 pulse width.
X = np.column_stack([
    df["v_bl"],
    np.log10(df["t_pw_us"])
])

# Targets: resistance and log10 device energy.
Y = np.column_stack([
    df["r_final_kohm"],
    np.log10(df["e_device_pj"])
])

if not np.isfinite(X).all() or not np.isfinite(Y).all():
    raise ValueError("Check the CSV for missing, invalid, or non-positive values.")

X_tr, X_te, Y_tr, Y_te = train_test_split(
    X, Y, test_size=0.20, random_state=42
)

print(f"Training: {len(X_tr)} rows | Testing: {len(X_te)} rows")


def models():
    return {
        "Linear": make_pipeline(
            StandardScaler(),
            LinearRegression()
        ),

        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            random_state=0,
            n_jobs=-1
        ),

        "GaussianProcess": make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(
                kernel=C(1.0) * RBF([1.0, 1.0]) + WhiteKernel(1e-4),
                normalize_y=True,
                random_state=0
            )
        )
    }


# Joint MLP: scale inputs AND each target using training data only.
print("Training joint MLP...", flush=True)

mlp_joint = TransformedTargetRegressor(
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

mlp_joint.fit(X_tr, Y_tr)

# Predictions are automatically returned in the original target units:
# column 0 = resistance in kOhm; column 1 = log10 energy in pJ.
predictions = {"MLP (joint, scaled)": mlp_joint.predict(X_te)}


# Other models are fitted separately for each target.
for name, unused_model in models().items():
    print(f"Training {name}...", flush=True)
    target_predictions = []

    for target in range(2):
        model = models()[name]
        model.fit(X_tr, Y_tr[:, target])
        target_predictions.append(model.predict(X_te))

    predictions[name] = np.column_stack(target_predictions)


# Evaluate resistance, log-energy, and energy in physical units.
rows = []

for name, pred in predictions.items():
    actual_values = [
        Y_te[:, 0],
        Y_te[:, 1],
        10.0 ** Y_te[:, 1]
    ]

    predicted_values = [
        pred[:, 0],
        pred[:, 1],
        10.0 ** pred[:, 1]
    ]

    target_names = [
        "Resistance (kOhm)",
        "Log10 energy (pJ)",
        "Energy (pJ)"
    ]

    for target, actual, estimated in zip(
        target_names, actual_values, predicted_values
    ):
        rows.append({
            "target": target,
            "model": name,
            "test_R2": r2_score(actual, estimated),
            "test_MAE": mean_absolute_error(actual, estimated)
        })

results = pd.DataFrame(rows)
print("\n", results.round(6).to_string(index=False))
results.to_csv(BASE / "ml_results.csv", index=False)


# Compare all models against the simulated test values.
colors = {
    "Linear": "tab:blue",
    "RandomForest": "tab:orange",
    "MLP (joint, scaled)": "tab:green",
    "GaussianProcess": "tab:red"
}

labels = [
    "Final resistance (kOhm)",
    "log10(device energy / pJ)"
]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for target, ax in enumerate(axes):
    actual = Y_te[:, target]
    plotted = [actual]

    for name, color in colors.items():
        estimated = predictions[name][:, target]
        plotted.append(estimated)

        ax.scatter(
            actual, estimated,
            s=14, alpha=0.6,
            color=color, label=name
        )

    values = np.concatenate(plotted)
    padding = 0.03 * (values.max() - values.min())
    limits = (values.min() - padding, values.max() + padding)

    ax.plot(limits, limits, "k--", linewidth=1, label="Ideal")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Simulated: " + labels[target])
    ax.set_ylabel("Predicted: " + labels[target])
    ax.set_title(f"20% test set ({len(X_te)} samples)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(BASE / "pred_vs_actual.png", dpi=300)
plt.savefig(BASE / "pred_vs_actual.pdf")

print(f"\nResults and plots saved in: {BASE}")
plt.show()
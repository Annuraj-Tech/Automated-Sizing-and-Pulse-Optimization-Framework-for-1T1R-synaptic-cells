from pathlib import Path
import json, time
import joblib
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
from sklearn.neighbors import NearestNeighbors
from sklearn.inspection import permutation_importance
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.metrics import r2_score, mean_absolute_error

BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUT = BASE / "forward_outputs"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42

df = pd.read_csv(BASE / "1t1r_dataset.csv")
need = ["w_mos_nm", "v_bl", "t_pw_us", "r_final_kohm", "e_device_pj", "w_init_nm"]
miss = [c for c in need if c not in df.columns]
if miss:
    raise ValueError(f"Missing columns: {miss}")
if not np.isfinite(df[need].to_numpy(float)).all():
    raise ValueError("CSV contains missing or invalid values.")
if (df["t_pw_us"] <= 0).any() or (df["e_device_pj"] <= 0).any():
    raise ValueError("Pulse width and energy must be positive.")
if not np.allclose(df["w_init_nm"], 1.5):
    raise ValueError("Models assume fixed w_init = 1.5 nm.")

df["log10_t_pw_us"] = np.log10(df["t_pw_us"])
FEATURE_SETS = {
    "2-input (V, t)": ["v_bl", "log10_t_pw_us"],
    "3-input (W, V, t)": ["w_mos_nm", "v_bl", "log10_t_pw_us"],
}
Y = np.column_stack([df["r_final_kohm"], np.log10(df["e_device_pj"])])

tr, te = train_test_split(np.arange(len(df)), test_size=0.20, random_state=SEED)
print(f"Training: {len(tr)} rows | Testing: {len(te)} rows", flush=True)


def make_model(name, n_features):
    if name == "Linear":
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "RandomForest":
        return RandomForestRegressor(300, random_state=0, n_jobs=-1)
    if name == "GaussianProcess":
        kernel = (C(1.0, (1e-3, 1e3)) * RBF([1.0] * n_features, (1e-2, 1e3))
                  + WhiteKernel(1e-4, (1e-8, 1e-1)))
        return make_pipeline(StandardScaler(), GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, random_state=0, n_restarts_optimizer=2))
    raise ValueError(name)


def make_joint_mlp():
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(64, 64), max_iter=5000, random_state=0)),
        transformer=StandardScaler())


def fit_predict(X_train, Y_train, X_test, n_features):
    preds = {}
    preds["MLP (joint, scaled)"] = make_joint_mlp().fit(X_train, Y_train).predict(X_test)
    for name in ["Linear", "RandomForest", "GaussianProcess"]:
        cols = []
        for t in range(2):
            m = make_model(name, n_features).fit(X_train, Y_train[:, t])
            cols.append(m.predict(X_test))
        preds[name] = np.column_stack(cols)
    return preds


rows, all_preds = [], {}
for fs_name, cols in FEATURE_SETS.items():
    print(f"\nTraining feature set: {fs_name}", flush=True)
    X = df[cols].to_numpy(float)
    preds = fit_predict(X[tr], Y[tr], X[te], len(cols))
    all_preds[fs_name] = preds
    for model_name, p in preds.items():
        pairs = [("Resistance (kOhm)", Y[te, 0], p[:, 0]),
                 ("Log10 energy (pJ)", Y[te, 1], p[:, 1]),
                 ("Energy (pJ)", 10 ** Y[te, 1], 10 ** p[:, 1])]
        for tname, a, e in pairs:
            rows.append({"feature_set": fs_name, "model": model_name, "target": tname,
                         "test_R2": r2_score(a, e), "test_MAE": mean_absolute_error(a, e)})

results = pd.DataFrame(rows)
print("\n", results.round(6).to_string(index=False))
results.to_csv(OUT / "forward_results.csv", index=False)

colors = {"Linear": "tab:blue", "RandomForest": "tab:orange",
          "MLP (joint, scaled)": "tab:green", "GaussianProcess": "tab:red"}
labels = ["Final resistance (kOhm)", "log10(device energy / pJ)"]

for fs_name, preds in all_preds.items():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for t, ax in enumerate(axes):
        actual = Y[te, t]
        vals = [actual]
        for name, color in colors.items():
            est = preds[name][:, t]
            vals.append(est)
            ax.scatter(actual, est, s=14, alpha=0.6, color=color, label=name)
        v = np.concatenate(vals)
        pad = 0.03 * (v.max() - v.min())
        lim = (v.min() - pad, v.max() + pad)
        ax.plot(lim, lim, "k--", lw=1, label="Ideal")
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Simulated: " + labels[t]); ax.set_ylabel("Predicted: " + labels[t])
        ax.set_title(f"{fs_name} | {len(te)} test samples")
        ax.grid(alpha=0.2); ax.legend(fontsize=8)
    fig.tight_layout()
    tag = "3input" if fs_name.startswith("3") else "2input"
    fig.savefig(OUT / f"pred_vs_actual_{tag}.png", dpi=300)
    plt.close(fig)

# Width sensitivity (held-out permutation importance, 3-input Gaussian process)
cols3 = FEATURE_SETS["3-input (W, V, t)"]
X3 = df[cols3].to_numpy(float)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
imp_rows = []
for t, ax in enumerate(axes):
    gp = make_model("GaussianProcess", 3).fit(X3[tr], Y[tr, t])
    imp = permutation_importance(gp, X3[te], Y[te, t], n_repeats=20,
                                 random_state=SEED, scoring="r2")
    ax.barh(["Width (nm)", "Voltage (V)", "log10 pulse (us)"],
            imp.importances_mean, xerr=imp.importances_std)
    ax.set_xlabel("Drop in held-out R2 when shuffled")
    ax.set_title(labels[t]); ax.grid(alpha=0.2)
    for n, m, s in zip(cols3, imp.importances_mean, imp.importances_std):
        imp_rows.append({"target": labels[t], "feature": n,
                         "importance_mean": m, "importance_std": s})
fig.tight_layout()
fig.savefig(OUT / "width_permutation_importance.png", dpi=300)
plt.close(fig)
pd.DataFrame(imp_rows).to_csv(OUT / "permutation_importance.csv", index=False)

# Deployment models: 3-input Gaussian processes refit on ALL rows
print("\nRefitting deployment models on all rows (takes a few minutes)...", flush=True)
gp_r = make_model("GaussianProcess", 3).fit(X3, Y[:, 0])
gp_e = make_model("GaussianProcess", 3).fit(X3, Y[:, 1])

resp_scaler = StandardScaler().fit(Y)
nn = NearestNeighbors(n_neighbors=2).fit(resp_scaler.transform(Y))
d, _ = nn.kneighbors(resp_scaler.transform(Y))
support_threshold = float(max(3.0 * np.quantile(d[:, 1], 0.95), 1e-6))

joblib.dump({
    "resistance_model": gp_r,
    "log_energy_model": gp_e,
    "feature_order": ["w_mos_nm", "v_bl", "log10(t_pw_us)"],
    "settings_min": X3.min(axis=0),
    "settings_max": X3.max(axis=0),
    "response_scaler": resp_scaler,
    "response_neighbours": nn,
    "support_threshold": support_threshold,
    "response_min": [float(df["r_final_kohm"].min()), float(df["e_device_pj"].min())],
    "response_max": [float(df["r_final_kohm"].max()), float(df["e_device_pj"].max())],
    "fixed_conditions": {"w_init_nm": 1.5, "r_init_kohm": 100.05,
                         "vwl_v": 1.5, "length_nm": 45},
}, OUT / "forward_models_3input.joblib")

with open(OUT / "forward_metadata.json", "w") as f:
    json.dump({"train_rows": int(len(tr)), "test_rows": int(len(te)), "seed": SEED,
               "note": "Deployment GPs refit on all rows; test scores use the 80/20 fit."},
              f, indent=2)

print(f"\nSaved to: {OUT}")
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import joblib
import numpy as np
import pandas as pd

from scipy.optimize import least_squares
from sklearn.neighbors import NearestNeighbors


BASE = Path(__file__).resolve().parent
OUT = BASE / "forward_outputs"

MODEL_PATH = OUT / "forward_models_3input.joblib"
CSV_PATH = BASE / "1t1r_dataset.csv"

# More starting points can improve the search but take longer.
N_STARTS = 6
MAX_EVALUATIONS = 100


class ForwardDesignPanel:
    def __init__(self, root):
        self.root = root
        self.root.title("1T1R Design — Saved Forward Models")
        self.root.geometry("1000x850")
        self.root.minsize(820, 650)

        self.messages = queue.Queue()
        self.ready = False
        self.busy = False

        self.r_var = tk.StringVar(value="144.2965")
        self.e_var = tk.StringVar(value="6485.78")
        self.r_tol_var = tk.StringVar(value="1.0")
        self.e_tol_var = tk.StringVar(value="5.0")
        self.status_var = tk.StringVar(
            value="Loading saved forward models..."
        )

        frame = ttk.Frame(root, padding=16)
        frame.pack(fill="both", expand=True)

        ttk.Label(
            frame,
            text="1T1R Forward-Model Design Assistant",
            font=("Segoe UI", 17, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            frame,
            text=(
                "Initial state = 1.5 nm | Initial resistance = 100.05 kOhm | "
                "Length = 45 nm | Word-line voltage = 1.5 V"
            ),
            wraplength=930,
        ).pack(anchor="w", pady=(6, 12))

        box = ttk.LabelFrame(
            frame,
            text="Desired response and acceptance tolerances",
            padding=12,
        )
        box.pack(fill="x")

        fields = [
            ("Target final resistance (kOhm)", self.r_var),
            ("Desired device energy (pJ)", self.e_var),
            ("Resistance tolerance (+/- kOhm)", self.r_tol_var),
            ("Energy tolerance (+/- %)", self.e_tol_var),
        ]

        for row, (label, variable) in enumerate(fields):
            ttk.Label(box, text=label).grid(
                row=row, column=0,
                sticky="w", padx=(0, 20), pady=5,
            )
            ttk.Entry(
                box, textvariable=variable, width=28
            ).grid(
                row=row, column=1, sticky="ew", pady=5
            )

        box.columnconfigure(1, weight=1)

        ttk.Label(
            box,
            text=(
                "Energy is a desired value, not a maximum budget. "
                "Width selection minimises target mismatch; "
                "it does not minimise transistor area."
            ),
            wraplength=880,
        ).grid(
            row=4, column=0, columnspan=2,
            sticky="w", pady=(8, 0),
        )

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=12)

        self.search_button = ttk.Button(
            buttons,
            text="Find configuration",
            command=self.start_search,
            state="disabled",
        )
        self.search_button.pack(side="left")

        ttk.Button(
            buttons,
            text="Clear results",
            command=lambda: self.display(""),
        ).pack(side="left", padx=10)

        ttk.Label(
            frame,
            textvariable=self.status_var,
            wraplength=930,
        ).pack(anchor="w", pady=(0, 8))

        result_frame = ttk.Frame(frame)
        result_frame.pack(fill="both", expand=True)

        self.result_text = tk.Text(
            result_frame,
            wrap="word",
            font=("Consolas", 10),
            padx=12,
            pady=12,
            state="disabled",
        )
        self.result_text.pack(
            side="left", fill="both", expand=True
        )

        scrollbar = ttk.Scrollbar(
            result_frame,
            command=self.result_text.yview,
        )
        scrollbar.pack(side="right", fill="y")

        self.result_text.configure(
            yscrollcommand=scrollbar.set
        )

        ttk.Label(
            frame,
            text=(
                "Predicted differences are not SPICE errors. "
                "A model-level pass requires fresh ngspice verification."
            ),
            foreground="#8B4513",
            wraplength=930,
        ).pack(anchor="w", pady=(10, 0))

        root.bind("<Return>", lambda event: self.start_search())

        self.root.after(100, self.poll_messages)

        threading.Thread(
            target=self.load_models,
            daemon=True,
        ).start()

    def display(self, text):
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.configure(state="disabled")

    def load_models(self):
        try:
            if not MODEL_PATH.is_file():
                raise FileNotFoundError(
                    f"Missing saved model:\n{MODEL_PATH}\n\n"
                    "Complete train_forward_width.py first."
                )

            if not CSV_PATH.is_file():
                raise FileNotFoundError(
                    f"Missing candidate/support CSV:\n{CSV_PATH}"
                )

            self.saved = joblib.load(MODEL_PATH)

            expected_order = [
                "w_mos_nm",
                "v_bl",
                "log10(t_pw_us)",
            ]

            if self.saved.get("feature_order") != expected_order:
                raise ValueError(
                    "Saved model feature order does not match "
                    "this panel."
                )

            self.r_model = self.saved["resistance_model"]
            self.e_model = self.saved["log_energy_model"]

            df = pd.read_csv(CSV_PATH)

            columns = [
                "w_mos_nm",
                "v_bl",
                "t_pw_us",
                "r_final_kohm",
                "e_device_pj",
            ]

            if any(c not in df.columns for c in columns):
                raise ValueError(
                    "CSV does not have the required columns."
                )

            if not np.isfinite(
                df[columns].to_numpy(dtype=float)
            ).all():
                raise ValueError("CSV contains invalid values.")

            if (
                (df["t_pw_us"] <= 0).any()
                or (df["e_device_pj"] <= 0).any()
            ):
                raise ValueError(
                    "Duration and energy must be positive."
                )

            self.settings = np.column_stack([
                df["w_mos_nm"],
                df["v_bl"],
                np.log10(df["t_pw_us"]),
            ])

            self.responses = np.column_stack([
                df["r_final_kohm"],
                np.log10(df["e_device_pj"]),
            ])

            self.lower = np.asarray(
                self.saved["settings_min"], dtype=float
            )
            self.upper = np.asarray(
                self.saved["settings_max"], dtype=float
            )
            self.span = self.upper - self.lower

            if (
                self.lower.shape != (3,)
                or self.upper.shape != (3,)
                or np.any(self.span <= 0)
            ):
                raise ValueError("Invalid saved settings bounds.")

            if (
                (self.settings < self.lower - 1e-8).any()
                or (self.settings > self.upper + 1e-8).any()
            ):
                raise ValueError(
                    "CSV settings do not match the saved model bounds. "
                    "Use the CSV used for training."
                )

            # Local settings-support heuristic.
            # This is NOT a proof of physical feasibility.
            self.scaled_settings = (
                self.settings - self.lower
            ) / self.span

            self.settings_nn = NearestNeighbors(
                n_neighbors=2
            ).fit(self.scaled_settings)

            distances, _ = self.settings_nn.kneighbors(
                self.scaled_settings
            )

            self.settings_support_limit = max(
                3.0 * np.quantile(distances[:, 1], 0.95),
                1e-6,
            )

            self.messages.put(("ready", None))

        except Exception as error:
            self.messages.put(("error", str(error)))

    def poll_messages(self):
        try:
            while True:
                kind, payload = self.messages.get_nowait()

                if kind == "ready":
                    self.ready = True
                    self.search_button.configure(state="normal")
                    self.status_var.set(
                        "Ready. Saved models loaded; no training."
                    )
                    self.display(
                        "Enter target resistance and desired device energy.\n\n"
                        "The optimiser searches width, voltage and duration\n"
                        "using the saved forward models.\n\n"
                        "Length and word-line voltage remain fixed.\n"
                        "Search time depends on model and computer speed."
                    )

                elif kind == "progress":
                    self.status_var.set(payload)

                elif kind == "result":
                    self.busy = False
                    self.search_button.configure(state="normal")
                    self.display(payload)
                    self.status_var.set("Search completed.")

                elif kind == "error":
                    self.busy = False
                    if self.ready:
                        self.search_button.configure(state="normal")
                    self.status_var.set("Operation could not complete.")
                    messagebox.showerror("Design panel", payload)

        except queue.Empty:
            pass

        self.root.after(100, self.poll_messages)

    @staticmethod
    def positive_number(variable, label):
        value = float(variable.get())
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                f"{label} must be a positive finite number."
            )
        return value

    def start_search(self):
        if not self.ready or self.busy:
            return

        try:
            target_r = self.positive_number(
                self.r_var, "Target resistance"
            )
            target_e = self.positive_number(
                self.e_var, "Desired energy"
            )
            r_tol = self.positive_number(
                self.r_tol_var, "Resistance tolerance"
            )
            e_tol_pct = self.positive_number(
                self.e_tol_var, "Energy tolerance"
            )

        except ValueError as error:
            messagebox.showerror("Invalid input", str(error))
            return

        self.busy = True
        self.search_button.configure(state="disabled")
        self.status_var.set(
            "Searching saved forward models — no retraining..."
        )

        threading.Thread(
            target=self.search,
            args=(target_r, target_e, r_tol, e_tol_pct),
            daemon=True,
        ).start()

    def search(self, target_r, target_e, r_tol, e_tol_pct):
        try:
            started = time.perf_counter()
            query = np.array([[target_r, np.log10(target_e)]])

            response_min = np.asarray(self.saved["response_min"])
            response_max = np.asarray(self.saved["response_max"])

            if (
                target_r < response_min[0]
                or target_r > response_max[0]
                or target_e < response_min[1]
                or target_e > response_max[1]
            ):
                raise ValueError(
                    "Requested resistance or energy lies outside "
                    "the observed response range."
                )

            scaled_query = self.saved[
                "response_scaler"
            ].transform(query)

            distance, _ = self.saved[
                "response_neighbours"
            ].kneighbors(scaled_query, n_neighbors=1)

            if distance[0, 0] > self.saved["support_threshold"]:
                raise ValueError(
                    "This resistance-energy pair is too far from "
                    "the observed response combinations according "
                    "to the support heuristic.\n\n"
                    "This does not prove physical infeasibility."
                )

            # Choose multiple starting points near the target response.
            # These seed the search; final settings may be new.
            response_score = (
                ((self.responses[:, 0] - target_r) / r_tol) ** 2
                + (
                    (10.0 ** self.responses[:, 1] - target_e)
                    / (target_e * e_tol_pct / 100.0)
                ) ** 2
            )

            indices = np.argsort(response_score)[
                :min(N_STARTS, len(self.settings))
            ]

            log_energy_scale = np.log10(
                1.0 + e_tol_pct / 100.0
            )

            def evaluate(z):
                setting = self.lower + z * self.span
                row = setting.reshape(1, -1)

                predicted_r = float(
                    self.r_model.predict(row)[0]
                )
                predicted_log_e = float(
                    self.e_model.predict(row)[0]
                )

                support_distance = float(
                    self.settings_nn.kneighbors(
                        z.reshape(1, -1),
                        n_neighbors=1,
                    )[0][0, 0]
                )

                return (
                    setting,
                    predicted_r,
                    predicted_log_e,
                    support_distance,
                )

            def residual(z):
                _, predicted_r, predicted_log_e, support_d = (
                    evaluate(z)
                )

                return np.array([
                    (predicted_r - target_r) / r_tol,
                    (
                        predicted_log_e - np.log10(target_e)
                    ) / log_energy_scale,
                    5.0 * max(
                        0.0,
                        support_d / self.settings_support_limit - 1.0,
                    ),
                ])

            candidates = []

            def add_candidate(z, solver_converged):
                setting, predicted_r, log_e, support_d = evaluate(z)

                with np.errstate(over="ignore"):
                    predicted_e = float(10.0 ** log_e)

                if not np.isfinite(
                    [predicted_r, predicted_e]
                ).all():
                    return

                r_error = abs(predicted_r - target_r)
                e_error_pct = (
                    abs(predicted_e - target_e)
                    / target_e * 100.0
                )

                supported = (
                    support_d <= self.settings_support_limit
                )

                feasible = (
                    supported
                    and r_error <= r_tol
                    and e_error_pct <= e_tol_pct
                )

                mismatch = (
                    (r_error / r_tol) ** 2
                    + (e_error_pct / e_tol_pct) ** 2
                )

                candidates.append({
                    "setting": setting,
                    "predicted_r": predicted_r,
                    "predicted_e": predicted_e,
                    "r_error": r_error,
                    "e_error_pct": e_error_pct,
                    "supported": supported,
                    "feasible": feasible,
                    "mismatch": mismatch,
                    "solver_converged": solver_converged,
                })

            for count, idx in enumerate(indices, start=1):
                self.messages.put((
                    "progress",
                    f"Searching starting point {count}/{len(indices)}...",
                ))

                z0 = (
                    self.settings[idx] - self.lower
                ) / self.span

                # Preserve seed as a candidate, then refine it.
                add_candidate(z0, False)

                result = least_squares(
                    residual,
                    x0=np.clip(z0, 1e-9, 1.0 - 1e-9),
                    bounds=(np.zeros(3), np.ones(3)),
                    max_nfev=MAX_EVALUATIONS,
                    ftol=1e-8,
                    xtol=1e-8,
                    gtol=1e-8,
                )

                add_candidate(result.x, bool(result.success))

            if not candidates:
                raise RuntimeError(
                    "Search produced no finite candidates."
                )

            # Prefer supported, feasible candidates.
            # Within that group, prefer smallest target mismatch.
            candidates.sort(key=lambda c: (
                not c["supported"],
                not c["feasible"],
                c["mismatch"],
            ))

            best = candidates[0]

            width, voltage, log_pw = best["setting"]
            pulse_us = 10.0 ** log_pw
            predicted_r = best["predicted_r"]
            predicted_e = best["predicted_e"]

            elapsed = time.perf_counter() - started

            if not best["supported"]:
                status = "REJECTED BY SETTINGS-SUPPORT HEURISTIC"
            elif best["feasible"]:
                status = "MEETS SURROGATE TARGET TOLERANCES"
            else:
                status = "NO TOLERANCE-MEETING CANDIDATE FOUND"

            text = (
                "CANDIDATE CONFIGURATION\n"
                "=======================\n"
                f"Transistor width:       {width:.3f} nm\n"
                "Transistor length:      45 nm (fixed)\n"
                "Word-line voltage:      1.5 V (fixed)\n"
                f"Bitline amplitude:      {voltage:.6f} V\n"
                f"Pulse duration:         {pulse_us:.6f} us\n\n"
                "FORWARD-MODEL PREDICTIONS\n"
                "=========================\n"
                f"Requested resistance:   {target_r:.4f} kOhm\n"
                f"Predicted resistance:   {predicted_r:.4f} kOhm\n"
                f"Signed difference:      "
                f"{predicted_r - target_r:+.4f} kOhm\n"
                f"Absolute difference:    "
                f"{best['r_error']:.4f} kOhm\n\n"
                f"Desired device energy:  {target_e:.3f} pJ\n"
                f"Predicted device energy:{predicted_e: .3f} pJ\n"
                f"Signed difference:      "
                f"{predicted_e - target_e:+.3f} pJ\n"
                f"Absolute difference:    "
                f"{best['e_error_pct']:.3f}%\n\n"
                f"Tolerances: +/-{r_tol:g} kOhm, "
                f"+/-{e_tol_pct:g}% energy\n"
                f"Settings-support check: "
                f"{'PASS' if best['supported'] else 'FAIL'}\n"
                f"STATUS: {status}\n\n"
                f"Search time: {elapsed:.2f} seconds\n"
                f"Starting points: {len(indices)}\n\n"
                "INTERPRETATION\n"
                "--------------\n"
                "Differences are computed using the same surrogate\n"
                "used for the search. They are not independent errors.\n"
                "Near-zero mismatch does not imply zero SPICE error.\n"
                "Support checks are heuristics, not safety guarantees.\n"
                "Width is a candidate, not a uniquely inferred size.\n"
                "No minimum-area or global-optimum claim is made.\n"
                "Use fresh ngspice simulation for verification."
            )

            record = pd.DataFrame([{
                "timestamp": pd.Timestamp.now().isoformat(),
                "target_r_kohm": target_r,
                "desired_energy_pj": target_e,
                "width_nm": width,
                "length_nm": 45.0,
                "vwl_v": 1.5,
                "v_bl": voltage,
                "t_pw_us": pulse_us,
                "predicted_r_kohm": predicted_r,
                "predicted_energy_pj": predicted_e,
                "r_error_kohm": best["r_error"],
                "energy_error_percent": best["e_error_pct"],
                "r_tolerance_kohm": r_tol,
                "energy_tolerance_percent": e_tol_pct,
                "supported_by_heuristic": best["supported"],
                "meets_surrogate_tolerances": best["feasible"],
                "search_seconds": elapsed,
            }])

            try:
                history = OUT / "forward_design_history.csv"
                record.to_csv(
                    history,
                    mode="a",
                    header=not history.exists(),
                    index=False,
                )

                record.to_csv(
                    OUT / "latest_forward_design.csv",
                    index=False,
                )

            except OSError as error:
                text += f"\n\nWarning: could not save CSV: {error}"

            self.messages.put(("result", text))

        except Exception as error:
            self.messages.put(("error", str(error)))


if __name__ == "__main__":
    root = tk.Tk()
    app = ForwardDesignPanel(root)
    root.mainloop()
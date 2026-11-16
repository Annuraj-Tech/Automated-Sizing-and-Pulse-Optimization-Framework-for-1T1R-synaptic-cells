import subprocess, os, csv, json, math, random, re, sys, shutil, tempfile, datetime

# ================= Configuration =================
NUM_SIMULATIONS = 1000
OUTPUT_CSV = "1t1r_dataset.csv"
PTM_FILE = None                  # or r"C:\full\path\to\ptm45.pm"
NMOS_MODEL = "nmos"              # must match .model name in ptm45.pm
SEED = 12345
VWL = 1.5                        # word-line voltage (V)
V_BL_RANGE = (0.8, 2.0)          # bit-line pulse amplitude (V)
DW_TARGET_RANGE_NM = (0.02, 1.4) # design target for state shift (log-uniform)
PW_LIMITS_S = (1e-6, 1e-2)       # pulse-width clamp
W_MOS_RANGE_NM = (200.0, 1200.0)
RANDOMIZE_INIT = False           # False keeps w_init = 1.5 nm (strict spec)
W_INIT_RANGE_NM = (1.0, 2.0)     # used only if RANDOMIZE_INIT
TMAX_DIV = 100                   # max timestep = pulse_width / TMAX_DIV
VERIFY_CONVERGENCE = True

# ================= VTEAM parameters (original, unmodified) =================
RON, ROFF = 100.0, 200000.0
VON, VOFF = -1.78, 0.0115
KON, KOFF = -8e-13, 8e-13
ALPHA_ON, ALPHA_OFF = 3, 3
W_ON, W_OFF, W_INIT = 0.0, 3e-9, 1.5e-9
NM = 1e9                         # unit conversion only: state node voltage = w in nm

NUM = r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?"
RE_W = re.compile(r"@@W_FINAL=\s*(" + NUM + ")")
RE_I = re.compile(r"@@IMAX=\s*(" + NUM + ")")
RE_ED = re.compile(r"^\s*e_dev\s*=\s*(" + NUM + ")", re.I | re.M)
RE_ES = re.compile(r"^\s*e_sup\s*=\s*(" + NUM + ")", re.I | re.M)

HEADER = ["sample_id", "w_mos_nm", "v_bl", "t_pw_us", "w_init_nm", "w_final_nm",
          "delta_w_nm", "r_init_kohm", "r_final_kohm", "e_device_pj",
          "e_supply_pj", "i_max_ua"]


def find_ngspice():
    for n in ("ngspice_con", "ngspice", "ngspice_con.exe", "ngspice.exe"):
        p = shutil.which(n)
        if p:
            return p
    sys.exit("ngspice executable not found on PATH")


def find_ptm():
    if PTM_FILE:
        if os.path.isfile(PTM_FILE):
            return os.path.abspath(PTM_FILE)
        sys.exit(f"PTM file not found: {PTM_FILE}")
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (here, os.path.dirname(here), os.getcwd()):
        p = os.path.join(d, "ptm45.pm")
        if os.path.isfile(p):
            return p
    sys.exit("ptm45.pm not found. Set PTM_FILE.")


NGSPICE_BIN = find_ngspice()
PTM_INC = find_ptm().replace("\\", "/")


def r_from_w_nm(w_nm):
    x = min(max((w_nm * 1e-9 - W_ON) / (W_OFF - W_ON), 0.0), 1.0)
    return RON + (ROFF - RON) * x          # linear VTEAM resistance mapping


def rate_m_per_s(v):
    return KOFF * (v / VOFF - 1.0) ** ALPHA_OFF if v > VOFF else 0.0


def build_netlist(tag, w_mos, v_bl, pw, w_init_m, tmax_div):
    td, tr = 0.1 * pw, 0.01 * pw
    t_stop = td + pw + 2 * tr + 0.2 * pw
    tstep, tmax = pw / 500.0, pw / tmax_div
    return f"""* 1T1R VTEAM run {tag}
.include "{PTM_INC}"
.options method=gear reltol=1e-4 vntol=1e-6 abstol=1e-12

.subckt vteam_model te be w_state
* Original VTEAM parameters (w in metres). State node voltage = w * NM (1 V = 1 nm).
.param Ron={RON} Roff={ROFF} Von={VON} Voff={VOFF} Kon={KON} Koff={KOFF} alpha_on={ALPHA_ON} alpha_off={ALPHA_OFF} w_on={W_ON} w_off={W_OFF} w_init={w_init_m:.6e} NM={NM:.1e}
Cw w_state 0 1.0 IC={{w_init*NM}}
Rleak w_state 0 1e15
B_dw 0 w_state I = {{NM}} * ( V(te)-V(be) > {{Voff}} ? (V(w_state) < {{w_off*NM}} ? {{Koff}} * pow((V(te)-V(be))/{{Voff}} - 1, {{alpha_off}}) : 0) : ( V(te)-V(be) < {{Von}} ? (V(w_state) > {{w_on*NM}} ? {{Kon}} * pow((V(te)-V(be))/{{Von}} - 1, {{alpha_on}}) : 0) : 0 ) )
B_mem te be I = (V(te)-V(be)) / ( {{Ron}} + ({{Roff}} - {{Ron}}) * ((V(w_state)/{{NM}} - {{w_on}}) / ({{w_off}} - {{w_on}})) )
.ends vteam_model

M1 mem_node wl_node 0 0 {NMOS_MODEL} l=45n w={w_mos}n
Xmem bl mem_node w_state vteam_model

Vwl wl_node 0 DC {VWL}
Vbl bl 0 PULSE(0 {v_bl} {td:.6e} {tr:.6e} {tr:.6e} {pw:.6e} {t_stop:.6e})

.control
  tran {tstep:.6e} {t_stop:.6e} 0 {tmax:.6e} uic
  let cell_curr = -i(Vbl)
  let p_dev = (v(bl) - v(mem_node)) * cell_curr
  let p_sup = v(bl) * cell_curr
  meas tran e_dev integ p_dev from=0 to={t_stop:.6e}
  meas tran e_sup integ p_sup from=0 to={t_stop:.6e}
  let w_end = v(w_state)[length(v(w_state)) - 1]
  let imax = maximum(abs(cell_curr))
  echo "@@W_FINAL= $&w_end"
  echo "@@IMAX= $&imax"
.endc
.end
"""


def run_case(tag, w_mos, v_bl, pw, w_init_m, workdir, tmax_div=TMAX_DIV):
    path = os.path.join(workdir, f"run_{tag}.cir")
    with open(path, "w") as f:
        f.write(build_netlist(tag, w_mos, v_bl, pw, w_init_m, tmax_div))
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    res = subprocess.run([NGSPICE_BIN, "-b", "-n", path], capture_output=True,
                         text=True, timeout=600, creationflags=flags, cwd=workdir)
    log = (res.stdout or "") + "\n" + (res.stderr or "")
    mw, mi = RE_W.search(log), RE_I.search(log)
    md, ms = RE_ED.search(log), RE_ES.search(log)
    if not (mw and mi and md and ms):
        with open(os.path.join(workdir, f"fail_{tag}.log"), "w") as f:
            f.write(log)
        errs = [l.strip() for l in log.splitlines()
                if re.search(r"error|could not|unknown|no such|fatal|singular|failed", l, re.I)]
        raise RuntimeError("ngspice output incomplete:\n    " +
                           "\n    ".join(errs[:8] or [l for l in log.splitlines() if l.strip()][-8:]))
    w_nm = float(mw.group(1))
    if not (-0.01 <= w_nm <= 3.05):
        raise RuntimeError(f"unphysical w_final={w_nm} nm")
    return {"w_final_nm": w_nm, "i_max": float(mi.group(1)),
            "e_dev": abs(float(md.group(1))), "e_sup": abs(float(ms.group(1)))}


def design_pulse(rng, v_bl, w0_nm):
    headroom = max(W_OFF * 1e9 - w0_nm - 0.1, 0.05)
    lo, hi = DW_TARGET_RANGE_NM[0], min(DW_TARGET_RANGE_NM[1], headroom)
    dw_t = math.exp(rng.uniform(math.log(lo), math.log(hi))) * 1e-9
    pw = dw_t / rate_m_per_s(v_bl)
    return min(max(pw, PW_LIMITS_S[0]), PW_LIMITS_S[1])


def verify_convergence(workdir):
    v, pw = 1.5, 0.5e-9 / rate_m_per_s(1.5)
    a = run_case("conv_a", 600.0, v, pw, W_INIT, workdir, tmax_div=50)
    b = run_case("conv_b", 600.0, v, pw, W_INIT, workdir, tmax_div=200)
    d0 = b["w_final_nm"] - W_INIT * 1e9
    err = abs(a["w_final_nm"] - b["w_final_nm"]) / max(abs(d0), 1e-12)
    print(f"Convergence check: dw(coarse)={a['w_final_nm']-1.5:.5f} nm, "
          f"dw(fine)={d0:.5f} nm, relative diff={err*100:.3f}%")
    if err > 0.005:
        sys.exit("Convergence check failed (>0.5%). Reduce TMAX_DIV step / reltol.")


def ngspice_version():
    try:
        r = subprocess.run([NGSPICE_BIN, "--version"], capture_output=True, text=True, timeout=20)
        return (r.stdout or r.stderr).strip().splitlines()[0]
    except Exception:
        return "unknown"


def main():
    rng = random.Random(SEED)
    workdir = tempfile.mkdtemp(prefix="ngspice_runs_")
    print(f"ngspice: {NGSPICE_BIN}\nPTM: {PTM_INC}\nWorkdir: {workdir}\n")
    if VERIFY_CONVERGENCE:
        verify_convergence(workdir)

    meta = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "ngspice": ngspice_version(), "ptm_file": PTM_INC, "nmos_model": NMOS_MODEL,
        "seed": SEED, "vteam": dict(Ron=RON, Roff=ROFF, Von=VON, Voff=VOFF, Kon=KON,
                                    Koff=KOFF, alpha_on=ALPHA_ON, alpha_off=ALPHA_OFF,
                                    w_on=W_ON, w_off=W_OFF, w_init=W_INIT, Cw_F=1.0),
        "notes": "State node stores w in nm (NM=1e9 unit conversion, no rate scaling). "
                 "Rectangular window. RESET-direction pulses only (device V > Voff). "
                 "Pulse width designed per sample from the VTEAM rate for target dw.",
        "v_bl_range": V_BL_RANGE, "dw_target_range_nm": DW_TARGET_RANGE_NM,
        "pw_limits_s": PW_LIMITS_S, "w_mos_range_nm": W_MOS_RANGE_NM,
        "vwl": VWL, "randomize_init": RANDOMIZE_INIT, "tmax_div": TMAX_DIV}
    with open(OUTPUT_CSV.replace(".csv", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    with open(OUTPUT_CSV, "w", newline="") as f:
        csv.writer(f).writerow(HEADER)

    ok = 0
    for i in range(1, NUM_SIMULATIONS + 1):
        w_mos = round(rng.uniform(*W_MOS_RANGE_NM), 1)
        v_bl = round(rng.uniform(*V_BL_RANGE), 3)
        w0_nm = round(rng.uniform(*W_INIT_RANGE_NM), 4) if RANDOMIZE_INIT else W_INIT * 1e9
        pw = design_pulse(rng, v_bl, w0_nm)
        try:
            r = run_case(i, w_mos, v_bl, pw, w0_nm * 1e-9, workdir)
            row = [i, w_mos, v_bl, round(pw * 1e6, 4), round(w0_nm, 4),
                   round(r["w_final_nm"], 5), round(r["w_final_nm"] - w0_nm, 5),
                   round(r_from_w_nm(w0_nm) / 1e3, 4),
                   round(r_from_w_nm(r["w_final_nm"]) / 1e3, 4),
                   round(r["e_dev"] * 1e12, 4), round(r["e_sup"] * 1e12, 4),
                   round(r["i_max"] * 1e6, 3)]
            with open(OUTPUT_CSV, "a", newline="") as f:
                csv.writer(f).writerow(row)
            ok += 1
            print(f"[{i}] V={v_bl} pw={pw*1e6:.2f}us -> w={r['w_final_nm']:.4f} nm | "
                  f"R={row[8]:.2f} kOhm | Edev={row[9]:.2f} pJ")
        except Exception as e:
            print(f"[{i}] FAILED: {e}")
    print(f"\nDone: {ok}/{NUM_SIMULATIONS} succeeded. Logs: {workdir}")


if __name__ == "__main__":
    main()
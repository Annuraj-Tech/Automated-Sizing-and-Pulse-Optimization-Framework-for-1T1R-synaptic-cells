import subprocess
import shutil
import sys
from pathlib import Path


# ============================================================
# EDIT ONLY THIS SECTION FOR EACH NEW CANDIDATE
# ============================================================

CANDIDATES = [
    # label, width_nm, bitline_voltage_V, pulse_duration_us,
    # target_resistance_kohm, target_energy_pJ

    ("145k_7000pj", 693.884, 1.560298, 347.047853, 145, 7000),
    ("point3",  699.134, 1.788833, 295.716374, 158, 7500),
]


# ============================================================
# DO NOT EDIT BELOW THIS LINE
# ============================================================

TEMPLATE = r'''* 1T1R verification: target R = @@R_TARGET@@ kOhm, desired energy = @@E_TARGET@@ pJ
* Candidate from forward-model optimisation
* State voltage represents w in nm: 1 V corresponds numerically to 1 nm

.include "ptm45.pm"

.options method=gear reltol=1e-4 vntol=1e-6 abstol=1e-12

.subckt vteam_model te be w_state

.param Ron=100 Roff=200000
.param Von=-1.78 Voff=0.0115
.param Kon=-8e-13 Koff=8e-13
.param alpha_on=3 alpha_off=3
.param w_on=0 w_off=3e-9 w_init=1.5e-9
.param NM=1e9

Cw w_state 0 1.0 IC={w_init*NM}
Rleak w_state 0 1e15

B_dw 0 w_state I = {NM} *
+ ( V(te)-V(be) > {Voff} ?
+   ( V(w_state) < {w_off*NM} ?
+     {Koff} * pow((V(te)-V(be))/{Voff}-1, {alpha_off}) : 0 )
+   :
+   ( V(te)-V(be) < {Von} ?
+     ( V(w_state) > {w_on*NM} ?
+       {Kon} * pow((V(te)-V(be))/{Von}-1, {alpha_on}) : 0 )
+     : 0 )
+ )

B_mem te be I = (V(te)-V(be)) /
+ ( {Ron} + ({Roff}-{Ron}) *
+   ((V(w_state)/{NM}-{w_on})/({w_off}-{w_on})) )

.ends vteam_model

M1 mem_node wl_node 0 0 nmos l=45n w=@@W_MOS@@n
Xmem bl mem_node w_state vteam_model

Vwl wl_node 0 DC 1.5

.param PW=@@PW_US@@u
.param TD={0.10*PW}
.param TR={0.01*PW}
.param TF={0.01*PW}
.param TSTOP={1.32*PW}

Vbl bl 0 PULSE(0 @@V_BL@@ {TD} {TR} {TF} {PW} {TSTOP})

.control
set numdgt=10
set nomoremode

let pulse_width = @@PW_US@@u
let stop_time = 1.32 * pulse_width
let output_step = pulse_width / 500
let maximum_step = pulse_width / 100

tran $&output_step $&stop_time 0 $&maximum_step uic

let cell_curr = -i(Vbl)
let cell_vol = v(bl) - v(mem_node)
let p_dev = cell_vol * cell_curr
let p_sup = v(bl) * cell_curr

meas tran e_dev integ p_dev from=0 to=$&stop_time
meas tran e_sup integ p_sup from=0 to=$&stop_time

let w_end_nm = v(w_state)[length(v(w_state))-1]

let r_end_kohm = (100 + (200000-100)*(w_end_nm/3))/1000
let e_dev_pj = e_dev * 1e12
let e_sup_pj = e_sup * 1e12
let imax_ua = maximum(abs(cell_curr)) * 1e6

let r_target = @@R_TARGET@@
let e_target = @@E_TARGET@@
let r_tolerance = 1
let e_tolerance_percent = 2

let r_signed_error = r_end_kohm - r_target
let r_abs_error = abs(r_signed_error)
let e_signed_error = e_dev_pj - e_target
let e_error_percent = 100 * abs(e_signed_error)/e_target

echo
echo "============= FRESH NGSPICE VERIFICATION ============="
echo "Candidate width:          @@W_MOS@@ nm"
echo "Candidate length:         45 nm"
echo "Word-line voltage:        1.5 V"
echo "Bitline amplitude:        @@V_BL@@ V"
echo "Pulse duration:           @@PW_US@@ us"
echo
echo "Final state (nm):         $&w_end_nm"
echo "Final resistance (kOhm):  $&r_end_kohm"
echo "Device energy (pJ):       $&e_dev_pj"
echo "Bitline supply energy(pJ): $&e_sup_pj"
echo "Peak bitline current(uA): $&imax_ua"
echo
echo "Target resistance(kOhm):  $&r_target"
echo "Signed R error (kOhm):    $&r_signed_error"
echo "Absolute R error(kOhm):   $&r_abs_error"
echo "Desired device energy(pJ): $&e_target"
echo "Signed energy error(pJ):  $&e_signed_error"
echo "Absolute energy error(%): $&e_error_percent"
echo

if w_end_nm < 0
  echo "INVALID RESULT: state below model range."
else
  if w_end_nm > 3
    echo "INVALID RESULT: state above model range."
  else
    if r_abs_error <= r_tolerance
      echo "Resistance check: PASS"
      if e_error_percent <= e_tolerance_percent
        echo "Energy check: PASS"
        echo "OVERALL: MEETS BOTH SPECIFICATIONS IN THIS SPICE RUN"
      else
        echo "Energy check: FAIL"
        echo "OVERALL: DOES NOT MEET BOTH SPECIFICATIONS"
      end
    else
      echo "Resistance check: FAIL"
      if e_error_percent <= e_tolerance_percent
        echo "Energy check: PASS"
      else
        echo "Energy check: FAIL"
      end
      echo "OVERALL: DOES NOT MEET BOTH SPECIFICATIONS"
    end
  end
end

echo "======================================================"

write verify_@@LABEL@@.raw v(bl) v(mem_node) v(w_state) cell_curr p_dev p_sup

@@FILE_REPORT@@

unset askquit
quit

.endc
.end
'''


def find_ngspice():
    path = shutil.which("ngspice.exe")
    if path:
        return path

    sys.exit("ngspice.exe GUI executable not found on PATH.")


def make_netlist(label, width_nm, voltage, duration_us, r_target, e_target):
    netlist = TEMPLATE

    replacements = {
        "@@LABEL@@": label,
        "@@W_MOS@@": f"{width_nm:.6f}",
        "@@V_BL@@": f"{voltage:.6f}",
        "@@PW_US@@": f"{duration_us:.6f}",
        "@@R_TARGET@@": f"{r_target:.6f}",
        "@@E_TARGET@@": f"{e_target:.6f}",
    }

    for old, new in replacements.items():
        netlist = netlist.replace(old, new)

    # Save the same displayed report using ngspice output redirection.
    start = netlist.index('echo "============= FRESH')
    stop = netlist.index("\nwrite verify_", start)
    report = netlist[start:stop]

    log_name = f"verify_{label}.log"

    report_lines = [
        f'echo "Fresh ngspice verification report" > {log_name}'
    ]

    for line in report.splitlines():
        if line.strip() == "echo":
            report_lines.append(f'echo " " >> {log_name}')
        elif line.lstrip().startswith("echo "):
            report_lines.append(f"{line} >> {log_name}")
        else:
            report_lines.append(line)

    netlist = netlist.replace(
        "@@FILE_REPORT@@",
        "\n".join(report_lines),
    )

    return netlist


def main():
    ngspice = find_ngspice()
    base_dir = Path(__file__).resolve().parent

    if not (base_dir / "ptm45.pm").is_file():
        sys.exit("Place ptm45.pm in the same folder as this Python file.")

    for item in CANDIDATES:
        label, width_nm, voltage, duration_us, r_target, e_target = item

        netlist = make_netlist(
            label, width_nm, voltage, duration_us, r_target, e_target
        )

        cir_file = base_dir / f"verify_{label}.cir"
        log_file = base_dir / f"verify_{label}.log"
        raw_file = base_dir / f"verify_{label}.raw"

        # Remove previous outputs so only fresh results are used.
        for old_file in (log_file, raw_file):
            if old_file.exists():
                old_file.unlink()

        cir_file.write_text(netlist, encoding="utf-8")

        print(f"\nRunning {cir_file.name} in the ngspice GUI...")

        result = subprocess.run(
            [ngspice, "-n", cir_file.name],
            cwd=str(base_dir),
        )

        if log_file.exists():
            print(log_file.read_text(errors="replace"))
        else:
            print(f"WARNING: No verification report generated for {label}")

        if result.returncode != 0:
            print(
                f"WARNING: ngspice returned code "
                f"{result.returncode} for {label}"
            )

        if not raw_file.exists():
            print(f"WARNING: No waveform file generated for {label}")

    print("\nAll verification runs finished.")


if __name__ == "__main__":
    main()
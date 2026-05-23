"""Run the inf-norm convex relaxation of the phase-only QCQP for a case.

Usage::

    python scripts/run_convex_relaxation.py --case case_1

Solves the SOCP

    min  t
    s.t. ||D_m w||^2 <= t,           m = 0..N-1
         Re(gain_coeffs . w) >= gmin
         ||w||_inf <= 1

with CVXPY + Mosek. Saves the relaxed weights and the surrogate
objective value to ``results/convex_relax_<case>.npz``.

Requires Mosek (academic license at mosek.com) and CVXPY.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from _common import CASES, build_setup                          # noqa: E402
import convex_relaxation                                          # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", default="case_1", choices=list(CASES))
    p.add_argument("--out-dir", default=os.path.join(_ROOT, "results"))
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--sub-sample", type=int, default=1)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[run_convex_relaxation] case = {args.case}", flush=True)
    setup = build_setup(args.case)
    ant = setup["ant"]
    data_iq = setup["data_iq"]
    N = ant.nEl

    data_power = np.abs(data_iq.get_fft_signal()) ** 2
    idx = ant.get_coupling_freq_idx(data_iq.freq_signal)

    print(f"[run_convex_relaxation] solving SOCP (N={N}) ...", flush=True)
    t0 = time.time()
    result = convex_relaxation.solve_convex_relaxation(
        S_f=ant.Sf, a_c=setup["a_c"], S_c=setup["S_c"],
        freq_S_in_signal_idx=idx,
        gmin=setup["gmin"],
        data_power=data_power,
        sub_sample=args.sub_sample,
        verbose=args.verbose,
    )
    elapsed = time.time() - t0

    t_star = result["t"]
    db_per_iter = 10.0 * np.log10(t_star)
    db_legacy = 10.0 * np.log10(t_star * 50.0 * N)
    print(f"\n[run_convex_relaxation] ====== relaxation result ({args.case}) ======")
    print(f"  total wall            : {elapsed:.1f}s")
    print(f"  status                : {result['status']}")
    print(f"  t (linear)            : {t_star:.6e}")
    print(f"  paper convention (dB) : {db_per_iter:+.4f}")
    print(f"  legacy convention (dB): {db_legacy:+.4f}")
    print(f"  gain achieved         : {result['gain']:.4f}  (gmin = {setup['gmin']:.4f})")
    print(f"=====================================================\n", flush=True)

    out_path = os.path.join(args.out_dir, f"convex_relax_{args.case}.npz")
    np.savez(out_path,
             w=result["w"], t=t_star, gain=result["gain"],
             status=result["status"], case=args.case)
    print(f"[run_convex_relaxation] saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()

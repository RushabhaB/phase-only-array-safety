"""Run the complex-Hermitian SDR via ADMM for a given paper case.

Usage::

    python scripts/run_sdr_admm.py --case case_1 [--max-iter 300] [--rho 1.0] \\
                                   [--alpha 1.0] [--warm-imm]

The script:
1. Builds the antenna setup, waveform, and per-port factor matrices D for
   the chosen case (see scripts/_common.py for case definitions);
2. (Optional) warm-starts the ADMM with the lifted iMM solution
   ``X = w_iMM w_iMM^H``;
3. Solves the SDR and prints the optimum t* in linear and dB units;
4. Saves the converged lifted matrix X*, the lifted vector w*, and the
   per-iteration history to ``results/sdr_admm_<case>.npz``.

Note on warm start
------------------
With cold start (X = I) the ADMM trajectory converges very slowly to the
SDR optimum -- empirically thousands of iterations and an unhelpful t
plateau. Warm-starting with the iMM solution reaches the optimum in a
few hundred iterations. The ``--warm-imm`` flag (default ON) does this
automatically by loading ``weights/imm/case_<N>_imm.mat``.
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

from _common import CASES, build_setup, load_imm_weights        # noqa: E402
import sdr_admm                                                  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", default="case_1", choices=list(CASES))
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=1.0,
                   help="ADMM over-relaxation (1.0 = none, 1.5 = common).")
    p.add_argument("--tol-p", type=float, default=1e-5)
    p.add_argument("--tol-d", type=float, default=1e-5)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--warm-imm", action="store_true", default=True,
                   help="Warm-start ADMM from the lifted iMM solution.")
    p.add_argument("--cold", dest="warm_imm", action="store_false",
                   help="Cold start (X = I); much slower convergence.")
    p.add_argument("--out-dir", default=os.path.join(_ROOT, "results"))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[run_sdr_admm] case = {args.case} ({CASES[args.case]['description']})",
          flush=True)
    print(f"[run_sdr_admm] building antenna setup ...", flush=True)
    setup = build_setup(args.case)
    N = setup["ant"].nEl
    D = setup["D"]
    gain_coeffs = setup["gain_coeffs"]
    gmin = setup["gmin"]
    print(f"[run_sdr_admm] N={N}, n_unique={setup['n_unique']}, "
          f"gmin={gmin:.2f}", flush=True)

    X_init = None
    t_init = None
    if args.warm_imm:
        try:
            w_imm = load_imm_weights(args.case)
            x_aug = np.concatenate([w_imm, np.array([1.0 + 0j])])
            X_init = np.outer(x_aug, x_aug.conj())
            DW = np.einsum("mki,i->mk", D, w_imm)
            t_init = float((np.abs(DW) ** 2).sum(axis=1).max())
            print(f"[run_sdr_admm] warm start from iMM, t_init={t_init:.6e}",
                  flush=True)
        except FileNotFoundError as e:
            print(f"[run_sdr_admm] warm start skipped: {e}", flush=True)

    print(f"[run_sdr_admm] solving ...", flush=True)
    t0 = time.time()
    result = sdr_admm.solve_sdr_admm(
        D=D, gain_coeffs=gain_coeffs, gmin=gmin,
        max_iter=args.max_iter, rho=args.rho, alpha=args.alpha,
        tol_p=args.tol_p, tol_d=args.tol_d,
        log_every=args.log_every,
        X_init=X_init, t_init=t_init,
        verbose=True,
    )
    elapsed = time.time() - t0

    t_star = result["t"]
    db_per_iter = 10.0 * np.log10(t_star)
    db_legacy = 10.0 * np.log10(t_star * 50.0 * N)
    print(f"\n[run_sdr_admm] ====== SDR result ({args.case}) ======")
    print(f"  iterations            : {result['iters']}")
    print(f"  total wall            : {elapsed:.1f}s")
    print(f"  t* (linear)           : {t_star:.6e}")
    print(f"  paper convention (dB) : {db_per_iter:+.4f}")
    print(f"  legacy convention (dB): {db_legacy:+.4f}")
    print(f"========================================\n", flush=True)

    out_path = os.path.join(args.out_dir, f"sdr_admm_{args.case}.npz")
    np.savez(out_path,
             X=result["X"], w=result["w"], t=t_star,
             iters=result["iters"], history=result["history"],
             case=args.case)
    print(f"[run_sdr_admm] saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()

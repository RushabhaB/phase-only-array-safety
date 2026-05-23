"""Run the inner Majorization-Minimization (iMM) solver for a paper case.

Usage::

    python scripts/run_imm.py --case case_1 [--max-iter 50000] [--device cuda]

iMM minimises the worst-case reflected power across the array under the
unit-modulus constraint via the closed-form dual update derived in §IV
of the paper. The full data cube
``S_data_cube_Vivaldi36.h5``  is required (not shipped with this
repository; see ``data/README.md`` for instructions on obtaining it).

The script saves the converged unit-modulus weights to
``results/imm_<case>.mat`` (a ``w_opt`` field compatible with the
warm-start path used by ``run_sdr_admm.py``).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import h5py
import scipy.io as sio

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from _common import CASES, build_setup                          # noqa: E402
import solver as _solver                                          # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", default="case_1", choices=list(CASES))
    p.add_argument("--max-iter", type=int, default=50000,
                   help="number of iMM iterations (50k is paper default).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                                                else "cpu",
                   choices=["cuda", "cpu"])
    p.add_argument("--cube",
                   default=os.path.join(_ROOT, "data", "S_data_cube_Vivaldi36.h5"),
                   help="path to the S-parameter data cube (16 GB hdf5).")
    p.add_argument("--out-dir", default=os.path.join(_ROOT, "results"))
    p.add_argument("--progress-every", type=int, default=500)
    args = p.parse_args()

    if not os.path.exists(args.cube):
        sys.exit(f"ERROR: S-cube not found at {args.cube}\n"
                 f"See data/README.md for instructions on obtaining it.")

    os.makedirs(args.out_dir, exist_ok=True)
    setup = build_setup(args.case)
    ant = setup["ant"]
    N = ant.nEl
    gmin = setup["gmin"]

    # Build the SolveArraySafety helper (uses the iMM entry point)
    solver_obj = _solver.SolveArraySafety(
        setup["S_c"], setup["a_c"], verbose=False, gmin=gmin)

    print(f"[run_imm] loading S-cube from {args.cube} ...", flush=True)
    t0 = time.time()
    with h5py.File(args.cube, "r") as hf:
        cube_np = hf["s_cube"][:]
    cube = torch.from_numpy(cube_np)
    del cube_np
    print(f"[run_imm] cube loaded in {time.time()-t0:.1f}s, "
          f"shape={tuple(cube.shape)}", flush=True)

    print(f"[run_imm] running iMM for {args.max_iter} iterations "
          f"on {args.device} ...", flush=True)
    t0 = time.time()
    w_gpu, record_obj, _, _, _ = solver_obj.solverArraySafety_phaseOnly(
        cube, max_iter=args.max_iter, debug=True,
        device=args.device, progress_every=args.progress_every,
        progress_label=f"iMM/{args.case}",
    )
    elapsed = time.time() - t0

    w_final = w_gpu.detach().cpu().numpy().astype(np.complex128)
    final_dB = float(record_obj[-1])
    print(f"\n[run_imm] ====== iMM result ({args.case}) ======")
    print(f"  iterations            : {args.max_iter}")
    print(f"  total wall            : {elapsed:.1f}s")
    print(f"  final Gamma^2 (legacy dB)   : {final_dB:+.4f}")
    print(f"  final Gamma^2 (paper dB)    : {final_dB - 10*np.log10(50.0 * N):+.4f}")
    print(f"  |w| min/max           : {np.abs(w_final).min():.6f} / "
          f"{np.abs(w_final).max():.6f}")
    print(f"=================================\n", flush=True)

    out_path = os.path.join(args.out_dir, f"imm_{args.case}.mat")
    sio.savemat(out_path, {"w_opt": w_final.reshape(-1, 1)})
    print(f"[run_imm] saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()

"""Build the per-port S-data cube and save it to HDF5.

The cube is the per-port Hermitian PSD object

    S_data_cube[p, r, q] = sum_n  |X(f_n)|^2 . S[p, r, n] . conj(S[p, q, n])

aggregated over all signal-frequency bins. It is universal (does NOT
depend on the beamforming weights or the steering direction), so it
only has to be built once and is reused across all cases by the iMM
solver in ``scripts/run_imm.py``.

This script saves the cube as ``data/S_data_cube_Vivaldi36.h5`` with a
single dataset ``s_cube`` of dtype ``complex64`` and shape
``(nEl, nEl, nEl) = (1296, 1296, 1296)`` for the default Vivaldi36
array (~16 GB on disk; the chunk shape is set so HDF5 reads slice-by-slice).

Requirements
------------
- ``data/Data/Smat_36x36_90MHz.mat`` is present (run
  ``scripts/fetch_data.sh`` first).
- ~32 GB RAM to hold the cube + intermediate batches.
- Runtime: ~5-30 minutes depending on CPU.

Usage
-----
::

    python scripts/build_cube.py [--batch-size 256] [--device cuda]
                                  [--out data/S_data_cube_Vivaldi36.h5]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import antenna_array as _antenna_array         # noqa: E402
import waveform as _waveform                   # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out",
                   default=os.path.join(_ROOT, "data", "S_data_cube_Vivaldi36.h5"),
                   help="output path for the HDF5 cube")
    p.add_argument("--batch-size", type=int, default=512,
                   help="frequency bins per batch (memory/speed knob)")
    p.add_argument("--device", default="cpu",
                   choices=["cpu", "cuda"],
                   help="cuda is faster but needs >16 GB VRAM")
    args = p.parse_args()

    out_path = os.path.abspath(args.out)
    if os.path.exists(out_path):
        sys.exit(f"ERROR: {out_path} already exists -- delete first")

    print(f"[build_cube] initialising Vivaldi36 array (loads Smat_36x36_90MHz.mat) ...",
          flush=True)
    ant = _antenna_array.Array(arrayName="Vivaldi36", num_segments=1)
    data_iq = _waveform.Waveform()
    N = ant.nEl
    print(f"[build_cube] N = {N}, Sf shape = {ant.Sf.shape}", flush=True)

    # The cube doesn't depend on the beamforming weights -- pick the
    # trivial broadside steering for the manifold argument.
    broadside_w = np.ones(N, dtype=np.complex64)
    _dummy_manifold = ant.calculate_array_manifold(
        data_iq.freq_signal, get_angle=True,
        theta_deg=[0], phi_deg=[0]
    )

    print(f"[build_cube] computing cube on {args.device} "
          f"(batch_size={args.batch_size}) ...", flush=True)
    t0 = time.time()
    _, _, s_cube = ant.calculate_reflected_power_directly(
        data_iq, _dummy_manifold, broadside_w,
        batch_size=args.batch_size, device=args.device,
        verbose=False, return_cube=True,
    )
    elapsed = time.time() - t0
    print(f"[build_cube] cube computed in {elapsed:.1f}s, "
          f"shape={tuple(s_cube.shape)}, dtype={s_cube.dtype}", flush=True)

    s_cube_np = s_cube.numpy()
    n_bytes = s_cube_np.nbytes
    print(f"[build_cube] cube memory: {n_bytes / 1e9:.2f} GB", flush=True)

    print(f"[build_cube] writing to {out_path} ...", flush=True)
    t0 = time.time()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as hf:
        hf.create_dataset(
            "s_cube",
            shape=s_cube_np.shape, dtype=s_cube_np.dtype,
            chunks=(256, s_cube_np.shape[1], s_cube_np.shape[2]),
            compression="gzip", data=s_cube_np,
        )
    print(f"[build_cube] wrote {out_path} in {time.time()-t0:.1f}s",
          flush=True)
    print(f"[build_cube] file size: "
          f"{os.path.getsize(out_path) / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()

"""Shared setup for the case-runner scripts.

Defines the three test cases used in the paper and a helper that builds
the antenna array, waveform, and the per-port factor matrices ``D``.
"""

import os
import sys
from typing import Sequence

import numpy as np

# add src/ to path so we can import the library modules from scripts/
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import antenna_array as _antenna_array        # noqa: E402
import waveform as _waveform                  # noqa: E402


# ---------------------------------------------------------------------------
# Case definitions (paper convention: Case 1 = 1 segment, Case 2 = 2 segs,
# Case 3 = 4 segs)
# ---------------------------------------------------------------------------
#
# theta_deg / phi_deg use the POLAR-angle convention employed in the
# `calculate_array_manifold` formula: u(theta, phi) =
# [cos(phi) sin(theta), sin(phi) sin(theta), cos(theta)]^T. The paper text
# loosely calls theta "elevation" but the formula and the values below treat
# it as the angle measured from the +z axis.
CASES = {
    "case_1": dict(n_segments=1,
                   theta_deg=[65],
                   phi_deg=[45],
                   description="full array, single steering direction"),
    "case_2": dict(n_segments=2,
                   theta_deg=[65, 45],
                   phi_deg=[145, 45],
                   description="two segments, beams crossing"),
    "case_3": dict(n_segments=4,
                   theta_deg=[65, 65, 65, 65],
                   phi_deg=[45, 135, -45, -135],
                   description="four segments, beams in each quadrant"),
}


def build_setup(case_key: str):
    """Build the antenna array, waveform, and per-port factor matrices D.

    Returns
    -------
    setup : dict with keys
        - ``ant``               : antenna_array.Array instance
        - ``data_iq``           : waveform.Waveform instance
        - ``S_c``               : (N, N) coupling matrix at center freq
        - ``a_c``               : (N,) steering manifold (complex)
        - ``D``                 : (N, n_unique, N) per-port factor matrices
        - ``gain_coeffs``       : (N,) row vector for Re(gain_coeffs . w) >= gmin
        - ``gmin``              : default gain lower bound = N/1.5
        - ``case``              : case definition dict
    """
    if case_key not in CASES:
        raise ValueError(f"unknown case '{case_key}'; choose from {list(CASES)}")
    cfg = CASES[case_key]

    ant = _antenna_array.Array(arrayName="Vivaldi36",
                               num_segments=cfg["n_segments"])
    data_iq = _waveform.Waveform()
    S_c = ant.get_center_freq_coupling_matrix(data_iq.fc)
    a_c = ant.calculate_array_manifold(
        data_iq.fc, get_angle=True,
        theta_deg=cfg["theta_deg"], phi_deg=cfg["phi_deg"]
    )
    a_c = np.asarray(a_c).ravel().astype(np.complex128)

    N = ant.nEl
    gmin = float(N / 1.5)

    # Per-port factor matrices D_m : Q_m = D_m^H D_m
    idx = ant.get_coupling_freq_idx(data_iq.freq_signal)
    K = len(idx)
    pwr = np.abs(data_iq.get_fft_signal()) ** 2
    s_idx_raw = np.asarray(idx).ravel()
    pwr_raw = np.asarray(pwr[:K]).ravel()
    unique_s_idx, inverse = np.unique(s_idx_raw, return_inverse=True)
    pwr_unique = np.bincount(inverse.ravel(), weights=pwr_raw,
                             minlength=len(unique_s_idx))
    pwr_sqrt = np.sqrt(pwr_unique).astype(np.float64)
    n_unique = len(unique_s_idx)
    D = np.empty((N, n_unique, N), dtype=np.complex128)
    for m in range(N):
        D[m] = ant.Sf[m, :, unique_s_idx] * pwr_sqrt[:, None]

    # Effective gain coefficient: Re(gain_coeffs . w) >= gmin
    I_minus_S = np.eye(N, dtype=np.complex128) - S_c
    gain_coeffs = (a_c.conj().reshape(1, -1) @ I_minus_S).ravel()

    return {
        "ant": ant, "data_iq": data_iq,
        "S_c": S_c, "a_c": a_c, "D": D,
        "gain_coeffs": gain_coeffs, "gmin": gmin,
        "case": cfg, "n_unique": n_unique,
    }


def load_imm_weights(case_key: str) -> np.ndarray:
    """Load the iMM-converged unit-modulus weights for the given case
    (used as a warm start for the SDR-ADMM solver). Returns a length-N
    complex array."""
    import scipy.io as sio
    path = os.path.join(_ROOT, "weights", "imm", f"{case_key}_imm.mat")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"iMM weights not found at {path}. They should ship with "
            f"the release; rerun the iMM solver to regenerate.")
    mat = sio.loadmat(path)
    return mat["w_opt"].ravel().astype(np.complex128)


def load_convex_relax_weights(case_key: str) -> np.ndarray:
    """Load the inf-norm convex-relaxation weights for the given case.
    These are NOT unit-modulus: ``|w_i| ≤ 1`` is the relaxed constraint.
    Returns a length-N complex array."""
    import scipy.io as sio
    path = os.path.join(_ROOT, "weights", "inf_norm",
                        f"{case_key}_convex_relax.mat")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"convex-relax weights not found at {path}. Run "
            f"scripts/run_convex_relaxation.py --case {case_key} to "
            f"regenerate.")
    mat = sio.loadmat(path)
    return mat["w_opt"].ravel().astype(np.complex128)

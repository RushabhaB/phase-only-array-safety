"""Convex relaxation of the phase-only QCQP for array safety.

Replaces the unit-modulus constraint ``|w_i| = 1`` with its convex hull
``||w||_∞ ≤ 1`` and solves the resulting SOCP exactly via CVXPY+Mosek.
This relaxation provides a tractable lower bound on the original
phase-only optimum.

Problem solved
--------------
::

    min  t
    s.t. ||D_m w||^2 ≤ t,           m = 0..N-1
         Re(gain_coeffs @ w) ≥ gmin
         ||w||_∞ ≤ 1

with the per-port factor ``D_m`` such that ``Q_m = D_m^H D_m`` is the
quadratic form appearing in the original objective.

The constraint is written as an SOC of dimension ``n_unique + 1`` per
port (where ``n_unique`` is the number of unique S-parameter frequency
bins after deduplication) rather than materialising the dense
``nEl × nEl`` Gram matrices ``Q_m``. This shrinks the Mosek problem by
roughly 10–25× compared to a naive lift.
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np


__all__ = ["solve_convex_relaxation"]


def solve_convex_relaxation(
    S_f: np.ndarray,
    a_c: np.ndarray,
    S_c: np.ndarray,
    freq_S_in_signal_idx: Sequence[int],
    gmin: float,
    *,
    data_power: Optional[np.ndarray] = None,
    sub_sample: int = 1,
    verbose: bool = False,
):
    """Solve the inf-norm convex relaxation of the array-safety QCQP.

    Parameters
    ----------
    S_f : (nEl, nEl, nFreqS) complex array
        Per-frequency coupling matrices.
    a_c : (nEl,) complex array
        Steering manifold at the center frequency.
    S_c : (nEl, nEl) complex array
        Coupling matrix at the center frequency (used to build the
        effective gain coefficient ``(I - S_c)^H a_c``).
    freq_S_in_signal_idx : sequence of int
        Indices into the last axis of ``S_f`` for each signal-frequency
        bin. Many signal frequencies may map to the same S-parameter
        bin; this is deduplicated internally.
    gmin : float
        Lower bound on the array gain.
    data_power : (K,) array, optional
        Per-bin PSD weights ``|X(f_k)|^2``. Defaults to all ones.
    sub_sample : int
        Stride into ``freq_S_in_signal_idx`` (for quick experiments;
        production runs should leave this at 1).
    verbose : bool
        Forward to Mosek's solver log.

    Returns
    -------
    result : dict
        ``{'w': complex array (nEl,), 't': float, 'gain': float,
           'status': str, 'solve_time_s': float}``
    """
    import cvxpy as cp                            # imported lazily

    nEl = S_f.shape[0]

    # ---- per-port factor matrices D_m ---------------------------------
    s_idx_raw = np.asarray(freq_S_in_signal_idx[::sub_sample]).ravel()
    K = len(s_idx_raw)
    if data_power is None:
        pwr_raw = np.ones(K)
    else:
        pwr_raw = np.asarray(data_power[::sub_sample][:K]).ravel()

    # Deduplicate: many signal freqs map to the same S-param bin.
    unique_s_idx, inverse = np.unique(s_idx_raw, return_inverse=True)
    pwr_unique = np.bincount(inverse.ravel(), weights=pwr_raw,
                             minlength=len(unique_s_idx))
    pwr_sqrt = np.sqrt(pwr_unique).astype(np.float64)
    n_unique = len(unique_s_idx)

    if verbose:
        print(f"[convex_relax] building {nEl} D_m of shape ({n_unique},"
              f" {nEl}) ...", flush=True)
    t0 = time.time()
    D_list = [S_f[m, :, unique_s_idx] * pwr_sqrt[:, None] for m in range(nEl)]
    if verbose:
        print(f"[convex_relax] D_m built in {time.time() - t0:.1f}s",
              flush=True)

    # ---- CVXPY problem -------------------------------------------------
    w = cp.Variable((nEl, 1), complex=True)
    t_var = cp.Variable(1, nonneg=True)

    constraints = [cp.sum_squares(D_list[m] @ w) <= t_var
                   for m in range(nEl)]

    I_minus_S = np.eye(nEl, dtype=np.complex128) - S_c
    gain_coeffs = a_c.conj().reshape(1, -1) @ I_minus_S       # (1, nEl)
    constraints.append(cp.real(gain_coeffs @ w) >= gmin)
    constraints.append(cp.norm(w, 'inf') <= 1)

    prob = cp.Problem(cp.Minimize(t_var), constraints)
    if verbose:
        print(f"[convex_relax] solving with MOSEK ...", flush=True)
    t0 = time.time()
    prob.solve(solver=cp.MOSEK, verbose=verbose)
    elapsed = time.time() - t0

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"convex relaxation did not converge: "
                           f"status={prob.status}")

    w_val = np.asarray(w.value).ravel()
    t_val = float(prob.value)
    gain = float((gain_coeffs @ w_val.reshape(-1, 1)).real)

    return {
        "w": w_val,
        "t": t_val,
        "gain": gain,
        "status": prob.status,
        "solve_time_s": elapsed,
    }

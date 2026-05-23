"""ADMM for the complex Hermitian SDR of the phase-only QCQP for array safety.

This module is a self-contained, library-style implementation of the
semidefinite relaxation (SDR) solver derived in the paper. It operates on
the (N+1) x (N+1) complex Hermitian matrix X directly using
``numpy.linalg.eigh`` for the PSD projection; no real-block expansion is
performed.

Problem solved
--------------
::

    min  t
    s.t. X ⪰ 0,                X ∈ H^{(N+1)x(N+1)}
         X_ii = 1,              i = 0..N
         trace(bar_Q_m X) ≤ t,  m = 0..N-1
         trace(A_g X) ≥ gmin

with bar_Q_m = blkdiag(Q_m, 0), Q_m = D_m^H D_m, and
A_g = (1/2) [[0, a], [a^H, 0]] for a = conj(gain_coeffs).

Cast as the standard conic program

    min c^T z
    s.t. A z = b
         z = (X, t, s_0..s_{N-1}, s_g) ∈ K

with K = H^{(N+1)}_+  ×  R_+  ×  R^N_+  ×  R_+, ADMM uses the splitting

    f(z) = c^T z + I_{A z = b}(z)            (affine projection)
    g(z̃) = I_K(z̃)                            (cone projection)

with consensus z = z̃. The KKT matrix for the affine projection has the
block structure

    [I_{N+1}        Q_diag                  0          ]
    [Q_diag^T       Gram + 1·1^T + I_N      0          ]
    [0              0                       ‖a‖²/2 + 1 ]

where ``Q_diag[i, m] = Q_m[i, i]`` for ``i < N`` (else 0) and
``Gram[m, m'] = trace(Q_m Q_{m'})``. The (μ, λ) block is Cholesky-factored
once at setup; per-iteration cost is one back-solve plus one Hermitian
eigendecomposition of size (N+1).

Public API
----------
- ``compute_gram(D)``               — N×N Gram matrix via rank-K_d trick
- ``solve_sdr_admm(D, gain_coeffs, gmin, ...)`` — main solver, returns
  the lifted matrix X*, the lifted vector w*, the SDR optimum t*, and
  per-iteration history.

References
----------
See ``docs/sdr_admm.pdf`` (in this repository) for the full derivation.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import scipy.linalg as la


__all__ = ["compute_gram", "solve_sdr_admm"]


# ---------------------------------------------------------------------------
# Gram matrix
# ---------------------------------------------------------------------------

def compute_gram(D: np.ndarray) -> np.ndarray:
    """Pairwise Gram matrix G[m, m'] = trace(Q_m Q_{m'}) via the
    rank-K_d decomposition Q_m = D_m^H D_m.

    Parameters
    ----------
    D : (N_ports, K_d, N_features) complex array
        Per-port factor matrices. ``Q_m`` is recovered as
        ``D[m].conj().T @ D[m]``; never materialised explicitly.

    Returns
    -------
    G : (N_ports, N_ports) real array
        ``G[m, m'] = ||D_m D_{m'}^H||_F^2 = trace(Q_m Q_{m'})``.
    """
    N_ports, K_d, n_feat = D.shape
    D_flat = D.reshape(N_ports * K_d, n_feat)
    Inner = D_flat @ D_flat.conj().T          # (N*K, N*K) complex
    I_sq = Inner.real ** 2 + Inner.imag ** 2  # real, |.|^2 entrywise
    del Inner
    return I_sq.reshape(N_ports, K_d, N_ports, K_d).sum(axis=(1, 3))


# ---------------------------------------------------------------------------
# ADMM setup (one-shot)
# ---------------------------------------------------------------------------

def _setup_kkt(D, gain_coeffs, gmin, port_idx):
    """Pre-compute the per-port Q diagonals, the Gram block, and the
    Cholesky factorization of the (μ, λ) KKT block. Returns a dict
    consumed by ``solve_sdr_admm``."""
    N = D.shape[2]
    M_aug = N + 1
    K_d = D.shape[1]
    a = gain_coeffs.astype(np.complex128)
    a_norm_sq = float((a.conj() @ a).real)

    P = len(port_idx)

    # Q_diag[i, m_idx] = Q_m[i, i] = sum_k |D[m, k, i]|^2  (real positive)
    Q_diag = np.zeros((M_aug, P), dtype=np.float64)
    for j, m in enumerate(port_idx):
        Q_diag[:N, j] = (np.abs(D[m]) ** 2).sum(axis=0)

    # Gram block
    if P == N:
        Gram = compute_gram(D)
    else:
        Gram = compute_gram(D[port_idx])

    gamma_coeff = a_norm_sq / 2.0 + 1.0

    # KKT (μ, λ) block, dense (M_aug + P) × (M_aug + P)
    n_KKT = M_aug + P
    KKT = np.empty((n_KKT, n_KKT), dtype=np.float64)
    KKT[:M_aug, :M_aug] = np.eye(M_aug)
    KKT[:M_aug, M_aug:] = Q_diag
    KKT[M_aug:, :M_aug] = Q_diag.T
    KKT[M_aug:, M_aug:] = Gram + 1.0 + np.eye(P)

    try:
        chol = la.cho_factor(KKT, lower=False, check_finite=False)
        solve_KKT = lambda b: la.cho_solve(chol, b, check_finite=False)
    except la.LinAlgError:
        lu = la.lu_factor(KKT, check_finite=False)
        solve_KKT = lambda b: la.lu_solve(lu, b, check_finite=False)

    return {
        "N": N, "M_aug": M_aug, "K_d": K_d, "P": P,
        "a": a, "gmin": gmin,
        "D_sel": np.ascontiguousarray(D[port_idx]),
        "D_sel_conj": np.ascontiguousarray(D[port_idx]).conj(),
        "Q_diag": Q_diag, "gamma_coeff": gamma_coeff,
        "solve_KKT": solve_KKT,
    }


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_sdr_admm(
    D: np.ndarray,
    gain_coeffs: np.ndarray,
    gmin: float,
    *,
    max_iter: int = 2000,
    rho: float = 1.0,
    alpha: float = 1.0,
    tol_p: float = 1e-5,
    tol_d: float = 1e-5,
    port_idx: Optional[list[int]] = None,
    X_init: Optional[np.ndarray] = None,
    t_init: Optional[float] = None,
    log_every: int = 20,
    verbose: bool = True,
):
    """Solve the complex Hermitian SDR via ADMM.

    Parameters
    ----------
    D : (N, K_d, N) complex array
        Per-port factor matrices with Q_m = D[m]^H D[m].
    gain_coeffs : (N,) complex array
        The linear functional of the gain constraint
        ``Re(gain_coeffs @ w) ≥ gmin`` (no conjugate; matches the form
        used in the iMM/PsGM solvers in this repository).
    gmin : float
        Lower bound on the array gain.
    max_iter : int
        Maximum ADMM iterations.
    rho : float
        ADMM penalty parameter.
    alpha : float
        Over-relaxation parameter in [1, 2). ``alpha = 1`` is no
        over-relaxation; ``1.5`` is a common default. We recommend 1.0
        for warm-started runs.
    tol_p, tol_d : float
        Primal and dual residual tolerances.
    port_idx : list[int] | None
        Subset of ports to include as trace constraints. Defaults to
        all N ports.
    X_init : (N+1, N+1) complex array | None
        Warm-start lifted matrix. Strongly recommended to pass the
        lifted iMM solution when available; cold start from I converges
        much more slowly.
    t_init : float | None
        Initial value of the SDR objective t. Ignored unless ``X_init``
        is provided.
    log_every : int
        Logging cadence (iterations).
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    result : dict
        ``{X, w, t, history, iters}`` where X is the converged lifted
        Hermitian matrix, w = X[:N, N] is the lifted vector, t the SDR
        optimum, and history a list of per-iteration diagnostics.
    """
    N = D.shape[2]
    M = N + 1
    if port_idx is None:
        port_idx = list(range(N))

    setup = _setup_kkt(D, gain_coeffs, gmin, port_idx)
    K_d = setup["K_d"]
    P = setup["P"]
    a = setup["a"]
    a_corner = a.conj()                     # b in A_g, ensures
                                            # trace(A_g X) = Re(a^T w)
    D_sel = setup["D_sel"]
    D_sel_conj = setup["D_sel_conj"]
    gamma_coeff = setup["gamma_coeff"]
    solve_KKT = setup["solve_KKT"]

    # ---- initialize ----
    if X_init is not None:
        X = X_init.copy().astype(np.complex128)
        X = 0.5 * (X + X.conj().T)
        if verbose:
            print(f"[sdr_admm] WARM START: |X_init|_F={np.linalg.norm(X):.3f}",
                  flush=True)
    else:
        X = np.eye(M, dtype=np.complex128)
    t = float(t_init) if t_init is not None else 0.0
    s = np.zeros(P)
    sg = 0.0
    ZX = X.copy()
    Zt = t
    Zs = s.copy()
    Zsg = sg
    YX = np.zeros_like(X)
    Yt = 0.0
    Ys = np.zeros(P)
    Ysg = 0.0
    inv_rho = 1.0 / rho

    history = []
    if verbose:
        print(f"[sdr_admm] starting (N={N}, P={P}, ρ={rho}, α={alpha}, "
              f"max_iter={max_iter})", flush=True)
        print(f"[sdr_admm] iter |   t        |   Zt        |  pri.res   "
              f"|  dual.res  |   min(eig)  | elapsed", flush=True)

    ZX_prev = ZX.copy()
    Zs_prev = Zs.copy()
    Zt_prev = Zt
    Zsg_prev = Zsg
    t_start = time.time()

    for k in range(max_iter):
        # ---- z-update (affine projection) ----
        Xref = ZX - YX / rho
        tref = Zt - Yt / rho
        sref = Zs - Ys / rho
        sgref = Zsg - Ysg / rho

        W_ref = Xref[:N, :N]
        # trace(Q_m W_ref) for m ∈ port_idx -- reshape+matmul is faster
        # than the equivalent einsum in NumPy.
        DW = (D_sel.reshape(P * K_d, N) @ W_ref).reshape(P, K_d, N)
        traces = np.einsum('mkj,mkj->m', DW, D_sel_conj).real

        w_ref = Xref[:N, M - 1]
        gain_lin = float((a @ w_ref).real)

        diag_real = Xref.diagonal().real
        rhs_mu = rho * (diag_real - 1.0)
        rhs_lam = rho * (traces - tref + sref) + 1.0
        rhs_gam = rho * (gain_lin - sgref - gmin)

        ml = solve_KKT(np.concatenate([rhs_mu, rhs_lam]))
        nu_mu = ml[:M]
        nu_lam = ml[M:]
        nu_gam = rhs_gam / gamma_coeff

        # ---- reconstruct z = z_ref - (c + A^T ν)/ρ ----
        X = Xref.copy()
        X[np.diag_indices(M)] -= nu_mu * inv_rho
        # ∑_m ν_λ_m Q_m  (over selected ports) via D_sel
        D_w = (D_sel * nu_lam[:, None, None]).reshape(P * K_d, N)
        W_update = D_sel_conj.reshape(P * K_d, N).T @ D_w
        X[:N, :N] -= W_update * inv_rho
        # A_g contributions (off-diagonal corners of X)
        a_half = (nu_gam * inv_rho * 0.5) * a_corner
        X[:N, M - 1] -= a_half
        X[M - 1, :N] -= a_half.conj()
        X = 0.5 * (X + X.conj().T)

        t = tref - inv_rho + nu_lam.sum() * inv_rho
        s = sref - nu_lam * inv_rho
        sg = sgref + nu_gam * inv_rho

        # ---- relaxation ----
        if alpha != 1.0:
            X_r = alpha * X + (1 - alpha) * ZX
            t_r = alpha * t + (1 - alpha) * Zt
            s_r = alpha * s + (1 - alpha) * Zs
            sg_r = alpha * sg + (1 - alpha) * Zsg
        else:
            X_r, t_r, s_r, sg_r = X, t, s, sg

        # ---- z̃-update (cone projection) ----
        M_psd = X_r + YX * inv_rho
        M_psd = 0.5 * (M_psd + M_psd.conj().T)
        M_psd = np.ascontiguousarray(M_psd)
        eigvals, eigvecs = np.linalg.eigh(M_psd)
        min_eig = float(eigvals.min())
        eigvals_pos = np.maximum(eigvals, 0.0)
        ZX_new = (eigvecs * eigvals_pos) @ eigvecs.conj().T
        ZX_new = 0.5 * (ZX_new + ZX_new.conj().T)

        Zt_new = max(t_r + Yt * inv_rho, 0.0)
        Zs_new = np.maximum(s_r + Ys * inv_rho, 0.0)
        Zsg_new = max(sg_r + Ysg * inv_rho, 0.0)

        # ---- dual update ----
        YX += rho * (X_r - ZX_new)
        Yt += rho * (t_r - Zt_new)
        Ys += rho * (s_r - Zs_new)
        Ysg += rho * (sg_r - Zsg_new)

        # ---- residuals ----
        pri_res = np.sqrt(
            np.linalg.norm(X - ZX_new) ** 2
            + (t - Zt_new) ** 2
            + np.linalg.norm(s - Zs_new) ** 2
            + (sg - Zsg_new) ** 2
        )
        dual_res = rho * np.sqrt(
            np.linalg.norm(ZX_new - ZX_prev) ** 2
            + (Zt_new - Zt_prev) ** 2
            + np.linalg.norm(Zs_new - Zs_prev) ** 2
            + (Zsg_new - Zsg_prev) ** 2
        )

        ZX_prev[:] = ZX_new
        Zs_prev[:] = Zs_new
        Zt_prev = Zt_new
        Zsg_prev = Zsg_new
        ZX, Zt, Zs, Zsg = ZX_new, Zt_new, Zs_new, Zsg_new

        if verbose and (k % log_every == 0 or k == max_iter - 1
                        or (pri_res < tol_p and dual_res < tol_d)):
            elapsed = time.time() - t_start
            print(f"[sdr_admm] {k:4d} | {t:+9.4e} | {Zt:+9.4e} | "
                  f"{pri_res:9.3e} | {dual_res:9.3e} | {min_eig:+9.3e} | "
                  f"{elapsed:7.1f}s", flush=True)

        history.append({
            "k": k, "t": t, "Zt": Zt,
            "pri_res": pri_res, "dual_res": dual_res, "min_eig": min_eig,
        })

        if pri_res < tol_p and dual_res < tol_d:
            if verbose:
                print(f"[sdr_admm] converged at iter {k}", flush=True)
            break

    w = ZX[:N, M - 1]
    return {
        "X": ZX, "w": w, "t": float(Zt),
        "iters": k + 1, "history": history,
    }

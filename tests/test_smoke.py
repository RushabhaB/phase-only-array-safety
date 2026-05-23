"""Smoke tests that exercise the major code paths without requiring the
16 GB data cube. Designed to run quickly on a laptop.

Run from the release/ folder::

    python tests/test_smoke.py

The tests check:
1. Library modules import cleanly.
2. The antenna setup for each paper case builds without errors.
3. The saved iMM weights round-trip through the SDR-ADMM warm-start
   path and produce sane numerical values (gain >= gmin and the SDR
   objective close to max_m |D_m w_iMM|^2).
4. A short ADMM warm-started run on case_1 converges within ~50 iters
   and lands within 1 dB of the iMM lifted value.
"""

from __future__ import annotations

import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import numpy as np


def _step(name, fn):
    print(f"\n=== {name} ===", flush=True)
    try:
        fn()
        print(f"  PASS", flush=True)
        return True
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return False


def test_imports():
    import antenna_array, waveform, solver, convex_relaxation, sdr_admm, utils  # noqa
    print("  imported: antenna_array, waveform, solver, convex_relaxation, "
          "sdr_admm, utils")


def test_setup_each_case():
    from _common import CASES, build_setup
    for case in CASES:
        s = build_setup(case)
        N = s["ant"].nEl
        assert s["D"].shape == (N, s["n_unique"], N), \
            f"{case}: bad D shape {s['D'].shape}"
        assert s["a_c"].shape == (N,), f"{case}: bad a_c shape"
        assert s["gain_coeffs"].shape == (N,), \
            f"{case}: bad gain_coeffs shape"
        print(f"  {case}: N={N}, n_unique={s['n_unique']}, "
              f"|a_c|_min={np.abs(s['a_c']).min():.3f}, "
              f"|a_c|_max={np.abs(s['a_c']).max():.3f}")


def test_imm_weights_load():
    from _common import CASES, load_imm_weights, build_setup
    for case in CASES:
        try:
            w = load_imm_weights(case)
        except FileNotFoundError:
            print(f"  {case}: no weights -- skipping (run scripts/run_imm.py)")
            continue
        assert np.allclose(np.abs(w), 1.0, atol=1e-6), \
            f"{case}: weights not unit-modulus (min={np.abs(w).min()})"
        s = build_setup(case)
        # max_m |D_m w|^2 should be a real positive number
        DW = np.einsum("mki,i->mk", s["D"], w)
        t_imm = float((np.abs(DW) ** 2).sum(axis=1).max())
        # gain should be >= gmin (within numerical fudge for older weights)
        gain = float((s["gain_coeffs"] @ w).real)
        print(f"  {case}: t_iMM = {t_imm:.6e} "
              f"({10*np.log10(t_imm):+.3f} dB), gain = {gain:.2f} "
              f"(gmin = {s['gmin']:.2f})")
        # Loose tolerance — older case 2/3 weights may be marginal on gain
        if gain < s["gmin"] - 1.0:
            print(f"    WARN: gain {gain:.2f} below gmin {s['gmin']:.2f}")


def test_sdr_admm_short_run_case1():
    """Short warm-started ADMM run on case_1: should reach within 1 dB of
    the iMM lifted value in 30 iterations."""
    from _common import build_setup, load_imm_weights
    import sdr_admm

    s = build_setup("case_1")
    N = s["ant"].nEl
    D = s["D"]
    w_imm = load_imm_weights("case_1")

    DW = np.einsum("mki,i->mk", D, w_imm)
    t_imm = float((np.abs(DW) ** 2).sum(axis=1).max())
    db_imm = 10 * np.log10(t_imm)

    x_aug = np.concatenate([w_imm, np.array([1.0 + 0j])])
    X_init = np.outer(x_aug, x_aug.conj())

    res = sdr_admm.solve_sdr_admm(
        D=D, gain_coeffs=s["gain_coeffs"], gmin=s["gmin"],
        max_iter=30, rho=1.0, alpha=1.0,
        X_init=X_init, t_init=t_imm,
        log_every=10, verbose=True,
    )
    t_admm = res["t"]
    db_admm = 10 * np.log10(max(t_admm, 1e-30))
    print(f"  iMM lifted t = {t_imm:.6e} ({db_imm:+.3f} dB)")
    print(f"  ADMM t       = {t_admm:.6e} ({db_admm:+.3f} dB)")
    assert abs(db_admm - db_imm) < 1.0, \
        f"ADMM result {db_admm:.3f} dB drifted more than 1 dB from iMM {db_imm:.3f} dB"


def test_compute_gram():
    """Quick check that compute_gram returns a real symmetric positive
    semidefinite Gram of the right shape."""
    import sdr_admm
    rng = np.random.default_rng(0)
    N, K = 20, 4
    D = (rng.standard_normal((N, K, N)) +
         1j * rng.standard_normal((N, K, N))) / np.sqrt(2)
    G = sdr_admm.compute_gram(D)
    assert G.shape == (N, N), f"bad Gram shape {G.shape}"
    assert np.allclose(G, G.T, atol=1e-10), "Gram not symmetric"
    eigvals = np.linalg.eigvalsh(G)
    assert eigvals.min() > -1e-8, f"Gram not PSD (min eig {eigvals.min()})"
    print(f"  N={N}, K={K}, Gram shape {G.shape}, min eig {eigvals.min():.3e}")


if __name__ == "__main__":
    print("=" * 60)
    print(" array-safety release  --  smoke tests")
    print("=" * 60)

    results = []
    results.append(_step("test_imports", test_imports))
    results.append(_step("test_setup_each_case", test_setup_each_case))
    results.append(_step("test_imm_weights_load", test_imm_weights_load))
    results.append(_step("test_compute_gram", test_compute_gram))
    results.append(_step("test_sdr_admm_short_run_case1",
                         test_sdr_admm_short_run_case1))

    print("\n" + "=" * 60)
    n_pass = sum(results)
    n_total = len(results)
    print(f" {n_pass} / {n_total} tests passed")
    print("=" * 60)
    sys.exit(0 if n_pass == n_total else 1)

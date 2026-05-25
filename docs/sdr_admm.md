# SDR ADMM Note

This release solves the complex Hermitian SDR directly in the lifted
matrix `X ∈ H^{(N+1)x(N+1)}`:

```text
min  t
s.t. X >= 0
     X_ii = 1
     trace(bar_Q_m X) <= t,  m = 0..N-1
     trace(A_g X) >= gmin
```

The implementation in [src/sdr_admm.py](../src/sdr_admm.py):

- builds `Q_m` implicitly from the thin factors `D_m` so the dense
  `Q_m = D_m^H D_m` matrices are never materialized,
- precomputes the pairwise Gram block
  `Gram[m,m'] = trace(Q_m Q_m') = ||D_m D_{m'}^H||_F^2`,
- factors the affine-projection KKT system once in `_setup_kkt()`,
- performs one affine projection and one PSD projection
  (`numpy.linalg.eigh`) per ADMM iteration,
- returns the lifted solution `X`, the extracted vector `w = X[:N, N]`,
  and the scalar optimum `t`.

The code uses the same gain convention as the rest of the release:
`Re(gain_coeffs @ w) >= gmin`, where `gain_coeffs = a_c^H (I - S_c)`.

For the fuller development history and derivation notes, see the original
research repository that this `release/` directory was extracted from.

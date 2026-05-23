# Phase-Only Beamforming under Array Safety Constraints

Reference implementation of the algorithms in:

> R. Balaji, R. Li, D. Cabric. *Phase-Only Beamforming under Array Safety
> Constraints in Extremely Large Antenna Arrays: A Majorization–Minimization
> Framework.* IEEE Transactions on Signal Processing, 2026.

This repository contains:

- the **inner Majorization–Minimization (iMM)** solver for the array-safety
  phase-only QCQP on the complex torus,
- the **projected sub-gradient (PsGM)** and **Riemannian sub-gradient (RsGM)**
  baselines that iMM is compared against in the paper,
- the **inf-norm convex relaxation** that provides the lower bound used in
  the optimality-gap discussion (§IV-D),
- a self-contained **complex Hermitian SDR via ADMM** that operates directly
  on the (N+1)×(N+1) Hermitian matrix and confirms iMM is essentially
  globally optimal for the three paper cases,
- reproduction scripts for **Case 1 / Case 2 / Case 3** as defined in the
  paper.

The 16 GB S-parameter data cube `S_data_cube_Vivaldi36.h5` is **not shipped
with this repository** (see [`data/README.md`](data/README.md)). The
unit-modulus iMM weights for each case (used to warm-start the ADMM-SDR and
to compare against the convex relaxation) are included in
[`weights/`](weights/).

---

## Quick start

```bash
# 1. environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. fetch the auxiliary data files (Smat ~517 MB; cube ~16 GB optional)
bash scripts/fetch_data.sh

# 3. solve the textbook Hermitian SDR via ADMM (warm-started from iMM)
#    -- only needs Smat, not the data cube; fastest demo (~3 min)
python scripts/run_sdr_admm.py --case case_1

# 4. solve the inf-norm convex relaxation (Mosek required; ~10 min)
python scripts/run_convex_relaxation.py --case case_1

# 5. (optional) run iMM from scratch -- needs the 16 GB cube
python scripts/run_imm.py --case case_1 --max-iter 50000
```

`--case case_2` and `--case case_3` exercise the other paper cases.

## Repository layout

```
release/
├── README.md                # this file
├── LICENSE                  # MIT
├── requirements.txt
├── src/                     # library modules
│   ├── antenna_array.py     # Vivaldi36 array geometry, S-matrix loader
│   ├── waveform.py          # LFM waveform generator
│   ├── solver.py            # SolveArraySafety class (iMM main entry point;
│   │                        # also hosts PsGM/RsGM via solver_FASTA(method_type=...))
│   ├── convex_relaxation.py # inf-norm SOCP via CVXPY+Mosek
│   ├── sdr_admm.py          # complex Hermitian SDR via ADMM (Sec. IV / Appx.)
│   └── utils.py
├── scripts/                 # reproduction drivers
│   ├── _common.py           # paper case definitions; antenna-setup helper
│   ├── run_imm.py           # iMM solver wrapper
│   ├── run_convex_relaxation.py
│   ├── run_sdr_admm.py
│   └── _main_legacy.py      # full research-code driver (verbose, kept for reference)
├── matlab/                  # MATLAB plotting / analysis scripts
│   ├── plot_uv_heatmaps.m
│   ├── AnalyzeArrayPerformance_Py.m
│   ├── beamform_coupling_matrix.m
│   └── arraySafetyDemo.m
├── tests/
│   └── test_smoke.py        # quick sanity checks (run without the data cube)
├── weights/
│   ├── imm/                 # unit-modulus iMM weights for each paper case
│   │   ├── case_1_imm.mat
│   │   ├── case_2_imm.mat
│   │   └── case_3_imm.mat
│   └── inf_norm/            # convex-relaxation (|w|_inf <= 1) weights
│       ├── case_1_convex_relax.mat
│       ├── case_2_convex_relax.mat
│       └── case_3_convex_relax.mat
├── data/                    # element coords + waveform shipped here
│   ├── Data/                # small files (PC_xyz_m, LFM IBW)
│   └── README.md            # how to fetch the larger Smat and S-cube
└── results/                 # solver outputs (created on demand)
```

## Paper case definitions

| Case | n_segments | (θ, φ) [degrees] | Description |
|---|---|---|---|
| Case 1 | 1 | (65°, 45°) | Full array, single steering direction. |
| Case 2 | 2 | (65°, 145°), (45°, 45°) | Two segments with beams crossing. |
| Case 3 | 4 | (65°, 45°), (65°, 135°), (65°, −45°), (65°, −135°) | Four segments, one beam per quadrant. |

**Convention.** The angles use the *polar* convention employed by the
`calculate_array_manifold` formula:
$\mathbf{u}(\theta, \phi) = [\cos\phi \sin\theta,\, \sin\phi \sin\theta,\, \cos\theta]^\top$.
This means θ is the angle measured from the +z axis (i.e., colatitude /
inclination), **not** elevation in the geographic sense. The paper text
uses "elevation" loosely; the numerical values above match the formula.

## Algorithm modules

### iMM (`src/solver.py`)

The primary algorithm of the paper. Entry point:
```python
import solver
sol = solver.SolveArraySafety(S_c, a_c, gmin=N/1.5)
w_opt, obj_history, *_ = sol.solverArraySafety_phaseOnly(
    S_data_cube, max_iter=50000, device='cuda'
)
```
Each iteration constructs a quadratic majorizer of the active constraints,
solves its dual via Mosek (closed-form on the simplex), and applies the
optimal phase-alignment update. See §IV of the paper.

### PsGM / RsGM (`src/solver.py`)

Sub-gradient baselines. Both are dispatched through `solver_FASTA` with
`method_type='PGD'` (Euclidean) or `'RGD_no_momentum'` / `'RGD_momentum'`
(Riemannian).

### Convex relaxation (`src/convex_relaxation.py`)

The inf-norm relaxation
$\min t \;\text{s.t.}\; \|D_m w\|^2 \le t,\;\;\mathrm{Re}(a^H w) \ge g_{\min},\;\;\|w\|_\infty \le 1$
formulated as a small SOCP (one SOC per port, dimension $n_\text{unique}+1$
rather than $n_\text{El}+1$) and solved via CVXPY+Mosek.

### Hermitian SDR via ADMM (`src/sdr_admm.py`)

The textbook complex Hermitian SDR

$$
\min t \;\text{ s.t. }\; X \succeq 0,\;\; X_{ii} = 1,\;\; \mathrm{trace}(\bar Q_m X) \le t,\;\; \mathrm{trace}(A_g X) \ge g_{\min}
$$

solved by ADMM directly on the (N+1)×(N+1) complex matrix
(`numpy.linalg.eigh` PSD projection — no real-block expansion). The KKT
matrix for the affine projection is Cholesky-factored once at setup; each
iteration is one back-solve plus one Hermitian eigendecomposition. With a
warm start from the iMM weights the algorithm converges in a few hundred
iterations. See `docs/sdr_admm.pdf` for the full derivation.

## Reproducing the paper

```bash
# All three cases. iMM step needs the data cube; the rest do not (they
# reuse the saved weights/optimized_results_iMM_case_*.mat).

for case in case_1 case_2 case_3; do
    python scripts/run_imm.py               --case $case --max-iter 50000
    python scripts/run_convex_relaxation.py --case $case
    python scripts/run_sdr_admm.py          --case $case --warm-imm
done
```

Expected outputs from `run_sdr_admm.py` warm-started from the iMM weights
(paper convention, $10\log_{10}(t^\star)$):

| Case | iMM (lifted) | SDR (ADMM) | Random feasible |
|---|---|---|---|
| Case 1 | −10.05 dB | −10.07 dB | ~ −9.9 dB |
| Case 2 | −10.98 dB | −11.01 dB | ~ −10.85 dB |
| Case 3 | −4.69 dB   | −4.72 dB | ~ −4.57 dB |

The SDR optimum is essentially equal to the iMM value in all three cases
(within ADMM residual error), and `X^*` is essentially rank-1 — strong
evidence that the iMM solution is at the global optimum of the phase-only
QCQP.

## Hardware / runtime notes

- **iMM**: built and tested on an RTX A6000 (single GPU), ~1 hour for
  50k iterations per case. CPU fallback works but is roughly 10× slower.
- **Convex relaxation**: Mosek interior-point on CPU; ~10 minutes per
  case, ~30 GB peak RAM.
- **ADMM-SDR**: CPU only (numpy `eigh` + scipy Cholesky). Warm-started,
  ~10 minutes per case at 300 iterations. ~ 4 GB RAM.

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use this code, please cite

```bibtex
@article{balaji2026phase,
  author  = {Balaji, Rushabha and Li, Ruifu and Cabric, Danijela},
  title   = {Phase-Only Beamforming under Array Safety Constraints in Extremely Large Antenna Arrays: A Majorization-Minimization Framework},
  journal = {IEEE Transactions on Signal Processing},
  year    = {2026},
}
```

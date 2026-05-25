# Data

This folder contains the small (sub-MB) data files needed to instantiate
the array geometry and load the LFM waveform. Two larger files are
**not** committed and have to be obtained separately:

## Shipped with the repo (`data/Data/`)

| File | Size | Purpose |
|---|---|---|
| `PC_xyz_m_36x36_Vivaldi.mat` | 1.4 KB | Element coordinates of the 36×36 Vivaldi array. Used by `antenna_array.Array._generate_element_coords`. |
| `LFM_1280MHz_IBW.mat` | 489 KB | Pre-generated LFM waveform samples (1.28 GHz IBW). Used by `waveform.Waveform.__init__`. |

## NOT shipped — must be obtained separately

Both files below are hosted as assets of a [GitHub Release][rel] of this
repository (free, up to 2 GB per asset, no LFS quotas).

| File | Size | SHA256 (first 12 chars) | Required by |
|---|---|---|---|
| `Data/Smat_36x36_90MHz.mat` | 517 MB | `4704b49dec5d…` | `antenna_array.Array._get_coupling_matrix` — i.e. **any** code instantiating `Array("Vivaldi36")`. Includes all paper-reproduction scripts. |
| `S_data_cube_Vivaldi36.h5` | 16 GB | (built locally) | `scripts/run_imm.py` only. **Build it from Smat with `python scripts/build_cube.py`** — takes ~10–30 min on CPU, ~32 GB RAM. No separate download needed. |

[rel]: https://github.com/RushabhaB/phase-only-array-safety/releases

### How to fetch

From the `release/` folder run

```bash
bash scripts/fetch_data.sh
```

The script downloads `Smat_36x36_90MHz.mat`, verifies its SHA256
checksum, and drops it in the right place
(`data/Data/Smat_36x36_90MHz.mat`). The cube download is disabled by
default; the supported workflow in this release is to build the cube
locally with `python scripts/build_cube.py` once `Smat` is present.

> Open `scripts/fetch_data.sh` and replace the `REPO_RELEASE_URL` and
> `CUBE_SHA256` placeholders with the release tag URL and the cube's
> SHA256 once you publish the GitHub Release.

### How to verify

```bash
sha256sum data/Data/Smat_36x36_90MHz.mat
# expected: 4704b49dec5d7e5735e73b5477ac4945c2ffc8d29a814b3ea40f7042191ae69b
```

## Obtaining the data

The data is available on request — please contact the corresponding
author. (The intended public hosting location, e.g. a Zenodo DOI, will be
linked here once it is set up.)

Once obtained, place `S_data_cube_Vivaldi36.h5` in this directory:

```
release/
└── data/
    └── S_data_cube_Vivaldi36.h5     ← 16 GB
```

`scripts/run_imm.py` looks for the cube at the default path
`data/S_data_cube_Vivaldi36.h5`; if you keep it elsewhere, pass
`--cube /path/to/S_data_cube_Vivaldi36.h5`.

## What you can run *without* the data cube

The following scripts do **not** require the data cube and will work as
soon as the Python dependencies are installed:

- `scripts/run_sdr_admm.py` — SDR-ADMM (uses the iMM weights in `weights/`
  to warm-start; does not touch the cube).
- `scripts/run_convex_relaxation.py` — inf-norm relaxation (uses
  per-port factor matrices `D_m` derived from the much smaller `ant.Sf`
  built inside `antenna_array.Array`).

The data cube is only needed by the iMM solver itself
(`scripts/run_imm.py`), because the iMM objective is evaluated against
the full per-element-per-frequency reflection coefficient cube.

# CEmulator examples

Open these notebooks from the repository root or `examples/`, using a Python
kernel with CEmulator installed (`pip install -e .` from the repository root).
NumPy, SciPy and Matplotlib are needed. CAMB and CLASS are optional except in
notebooks that explicitly compare against those solvers.

For small GP predictions, set `MKL_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and
`OPENBLAS_NUM_THREADS=1` before starting Python or Jupyter.

| Example | Contents | Additional dependencies |
| --- | --- | --- |
| `comoving-distance.ipynb` | Background distances and neutrino treatments | None |
| `pklin.ipynb` | Linear matter power; CAMB/CLASS comparison | CAMB, CLASS for comparison cells |
| `pkmm.ipynb` | Nonlinear matter power, HMCODE2020 and halofit | CAMB, CLASS for comparison cells |
| `sigma8.ipynb` | Total-matter/cb sigma8 and neutrino splitting | CAMB, CLASS for comparison cells |
| `hmf.ipynb` | Differential and cumulative halo mass functions | None |
| `pkhm.ipynb` | Halo–matter power by density and mass threshold | None |
| `ximm.ipynb` | Matter correlation functions | None |
| `xihm.ipynb` | Halo–matter correlations in mass bins | None |
| `xihh.ipynb` | Halo–halo threshold and mass-bin correlations | None |
| `GalaxyEmulator.ipynb` | HOD, galaxy correlations, wp and DeltaSigma | None |
| `Cell.ipynb` | Convergence C_ell for a source plane or source distribution | None |
| `SpectralEquivalence.ipynb` | Auxiliary dark-energy models and amplitude matching | CAMB, CLASS |
| `example_hod_usage.py` | Executable HOD examples, including lensing components | None |

Run the script from the repository root with
`python examples/example_hod_usage.py` after installing CEmulator.

Use increasing redshift, mass and log10 number-density arrays. Halo–matter,
halo–halo and galaxy examples use z <= 0.8. Distances are comoving; individual
examples label the units. The weak-lensing example documents its source and
scale limitations.

Notebooks are stored without execution output. Machine-specific simulation
comparisons and checkpoints have been removed; the original examples are
backed up separately in `/home/chenzhao/csst/simulation/csstemu-test-nb/`.

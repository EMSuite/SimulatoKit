# tools

Scripts that generate the test fixtures, measure the benchmarks and check the public API. Run them from the repo root.

## Python environment

The fixture generators and the abTEM side of the benchmarks need the pinned reference, abTEM 1.0.10, with the NumPy,
SciPy and ASE versions the committed fixtures were made with (`tools/requirements.txt`). From a fresh clone:

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r tools/requirements.txt
```

`.venv` is ignored by git. Python 3.12 and 3.14 give byte-identical fixture arrays; only the `python` version stamp in a
`*-pins.json` differs.

## Fixture generators (`pins/`)

Each `pins/<subject>/pins.py` runs abTEM 1.0.10 in float64 (NumPy FFT, eager), checks it against the published formulas
where the script says so, and writes its fixtures in place under `Tests/SimulatoKitParityTests/Fixtures/`. A failed check
stops the script before anything is written. The inventory of every fixture and its generator is
`Tests/SimulatoKitParityTests/Fixtures/README.md`.

```sh
tools/pins/regenerate.sh                      # every generator, in order (PYTHON=<interpreter> to use another venv)
.venv/bin/python -I -B tools/pins/compare.py  # judge what moved: equal, float noise, or DIFFERS
git checkout -- Tests Sources                 # put the committed references back after a check
```

The Linux reference job (`.github/workflows/references.yml`, weekly and on every PR that touches `tools/pins/`, the
requirements or the fixtures) runs exactly these two commands on GitHub's Ubuntu runner and fails on any DIFFERS. The
committed fixtures were written on a Mac; Linux reproduces them to float noise (about 1e-15 relative), which
`compare.py` tells apart from a real change (`|a - b| <= 1e-12 + 1e-9 max(|a|, |b|)`; float32 bounds for the arrays
`tools/pins/float32-arrays.txt` names, each with its reason). `tools/pins/test_compare.py` tests the comparison itself.

| Generator | Writes |
|---|---|
| `pins/potential-multislice/pins.py` | `slice-1-pins.json`, `slice-1-arrays.bin`, `Sources/SimulatoMultislice/LobatoTable.swift` |
| `pins/stem-detectors/pins.py` | `slice-2-pins.json`, `slice-2-arrays.bin` (imports `potential-multislice`) |
| `pins/ctf/pins.py` | `slice-3-pins.json`, `slice-3-arrays.bin` (imports `potential-multislice`) |
| `pins/ctf-profiles/pins.py` | `ctf-profiles-pins.json` (imports `ctf`'s formulas) |
| `pins/hrtem-series/pins.py` | `hrtem-series-pins.json`, `hrtem-series-arrays.bin` (imports `potential-multislice` and `ctf`) |
| `pins/sampling-tilt/pins.py` | `slice-4-pins.json`, `slice-4-arrays.bin` |
| `pins/scan-patterns/pins.py` | `slice-4-scan-pins.json`, `slice-4-scan-arrays.bin` |
| `pins/waves/pins.py` | `waves-pins.json`, `waves-arrays.bin` |
| `pins/linescan/pins.py` | `linescan-pins.json`, `linescan-arrays.bin` |
| `pins/icom/pins.py` | `icom-pins.json`, `icom-arrays.bin` |
| `pins/slab/pins.py` | `slab-pins.json`, `slab-arrays.bin` |
| `pins/beam-intensities/pins.py` | `beam-intensities-pins.json` (imports `slab`'s rules) |
| `pins/kirkland/pins.py` | `kirkland-pins.json`, `kirkland-arrays.bin`, `Sources/SimulatoMultislice/KirklandTable.swift` (reads `slab-pins.json`, so runs after `slab`) |
| `pins/thermal-sites/pins.py` | `thermal-sites-pins.json`, `thermal-sites-arrays.bin` (imports `potential-multislice`) |
| `pins/flexible-annular/pins.py` | `flexible-annular-pins.json`, `flexible-annular-arrays.bin` (imports `potential-multislice`, `stem-detectors`, `beam-intensities`) |
| `pins/cbed-series/pins.py` | `cbed-series-pins.json`, `cbed-series-arrays.bin` (imports `potential-multislice`, `stem-detectors`, `beam-intensities`) |
| `pins/hrtem-si110/pins.py` | `hrtem-si110-pins.json`, `hrtem-si110-arrays.bin` (imports `potential-multislice`, `ctf`, `beam-intensities`) |
| `pins/potential-project-tile/pins.py` | none: prints the numbers `SimulatoMultisliceTests/PotentialProjectTileTests.swift` holds inline |
| `pins/polar/pins.py` | none: prints the numbers `SimulatoDetectorTests/PolarMeasurementTests.swift` holds inline |
| `pins/crystal-slices/pins.py` | none: prints the numbers `SimulatoMultisliceTests/CrystalSliceThicknessTests.swift` holds inline |
| `pins/image-interpolate/pins.py` | none: prints the numbers `SimulatoCoreTests/ImageInterpolationTests.swift` holds inline |
| `pins/image-gaussian/pins.py` | none: prints the numbers `SimulatoCoreTests/ImageGaussianFilterTests.swift` holds inline |
| `pins/simulation-plan/pins.py` | none: prints the numbers `SimulatoMultisliceTests/SimulationPlanTests.swift` holds inline |

## Other tools

- `benchmarks/`: the per-function table of `docs/benchmarks.md`; `docs/benchmarks.md` gives the commands. Output goes
  to `benchmarks/out/` (ignored by git).
- `consumer-check/`: a separate package that compiles against the public API (`tools/consumer-check/run.sh`, run by CI).
- `metal-fft-gain/`: the measurements behind the Metal FFT round-trip bounds (see its README).
- `build-docs.sh`: builds the DocC site (run by the docs workflow).

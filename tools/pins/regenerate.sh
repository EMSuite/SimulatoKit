#!/bin/bash
# Rebuild every committed reference of SimulatoKit from the pinned abTEM environment, in place. Run from the repo root
# with the venv of tools/README.md (or PYTHON=<interpreter>); then `python tools/pins/compare.py` says whether anything
# moved beyond float noise. The Linux reference job (.github/workflows/references.yml) runs exactly this.
set -euo pipefail
PYTHON=${PYTHON:-.venv/bin/python}

# Generators that write fixture files (tools/README.md lists what each writes). stem-detectors, ctf and thermal-sites import
# potential-multislice, which therefore runs first; hrtem-series imports both potential-multislice and ctf.
writers=(potential-multislice stem-detectors ctf ctf-profiles hrtem-series sampling-tilt scan-patterns waves icom linescan slab beam-intensities kirkland thermal-sites flexible-annular cbed-series)
# Generators that print the numbers a test holds inline. They run so a broken reference shows here; their output goes to
# the log and is not compared (the numbers live in the Swift test files).
printers=(potential-project-tile polar crystal-slices image-crop-difference image-interpolate image-gaussian azimuthal linescan-defaults simulation-plan)

for g in "${writers[@]}"; do
  echo "::group::$g (writes fixtures)"
  "$PYTHON" -I -B "tools/pins/$g/pins.py"
  echo "::endgroup::"
done
for g in "${printers[@]}"; do
  echo "::group::$g (prints inline numbers)"
  "$PYTHON" -I -B "tools/pins/$g/pins.py"
  echo "::endgroup::"
done

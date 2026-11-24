"""SimulatoKit simulation plan: the inline pins.

abTEM 1.0.10: for each case (extent, energy, largest angle), the sampling rule min(0.05 Å, λ / (3 β)), abTEM's gpts for
that sampling (ceil(extent / sampling)), the same rounded up to a length with factors 2, 3 and 5 only, and abTEM's
`cutoff_angles` on the rounded grid. Prints the numbers SimulationPlanTests holds inline; writes nothing.

Run from the repo root:  .venv/bin/python -I -B tools/pins/simulation-plan/pins.py
"""

from __future__ import annotations

import math

import abtem
from abtem.core.energy import energy2wavelength


def friendly(n):
    m = n
    while True:
        k = m
        for p in (2, 3, 5):
            while k % p == 0:
                k //= p
        if k == 1:
            return m
        m += 1


CASES = [  # extent (Å), energy (eV), largest angle (mrad)
    ((10.86, 15.36), 300e3, 20.0),   # Si [110] 2 x 4 units, HRTEM aperture: sampling 0.05 wins
    ((20.0, 20.0), 200e3, 200.0),    # STEM outer angle 200 mrad: λ / (3 β) wins
    ((25.0, 31.0), 80e3, 150.0),     # 80 kV
    ((40.1, 40.1), 300e3, 90.0),
]

for extent, energy, beta in CASES:
    lam = energy2wavelength(energy)
    sampling = min(0.05, lam / (3 * beta * 1e-3))
    wave = abtem.PlaneWave(energy=energy, extent=extent, sampling=sampling)
    gpts = wave.gpts
    rounded = tuple(friendly(n) for n in gpts)
    grid = abtem.PlaneWave(energy=energy, extent=extent, gpts=rounded)
    print(f"extent {extent} energy {energy} beta {beta}: lambda {lam!r} sampling {sampling!r} abtem gpts {gpts} "
          f"rounded {rounded} cutoff_angles {tuple(float(a) for a in grid.cutoff_angles)!r}")

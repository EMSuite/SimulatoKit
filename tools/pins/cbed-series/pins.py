"""SimulatoKit CBED thickness series: the pin generator.

Runs abTEM 1.0.10 in float64 (numpy FFT, eager): a 10 mrad probe at 200 kV, placed at one position of Si [110] (the kit's
slab) and of Al [001], multislice through the crystal with an exit plane at 20, 50 and 100 nm, and the pixelated
detector (`PixelatedDetector(max_angle="cutoff")`) on each exit wave. Each pattern is re-derived in NumPy (the potentials
of one period tiled, text_multislice, text_probe, text_pixelated) and asserted equal to abTEM's at 1e-10 of the maximum.
A failure stops the generator. Writes

  Tests/SimulatoKitParityTests/Fixtures/cbed-series-pins.json
  Tests/SimulatoKitParityTests/Fixtures/cbed-series-arrays.bin   (float64 little-endian)

Atoms come from snapped coordinates (abTEM 1.0.10's infinite-projection bug, as tools/pins/slab). The planned two-beam
CBED thickness of DiffractoKit #170 reads the central disc of these series.

Run from the repo root:  .venv/bin/python -I -B tools/pins/cbed-series/pins.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIXTURES = os.path.join(ROOT, "Tests", "SimulatoKitParityTests", "Fixtures")


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "tools", "pins", name, "pins.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s1 = load("potential-multislice")  # sets abTEM's configuration (float64, numpy FFT, eager)
s2 = load("stem-detectors")
bi = load("beam-intensities")

import abtem  # noqa: E402
import ase  # noqa: E402
from ase.build import bulk  # noqa: E402

ENERGY = 200e3
CUTOFF = 10.0  # mrad
SAMPLING = 0.1  # A, about
TARGET_NM = (2, 5, 10)  # tens of nm below: 20, 50 and 100 nm
BLOB = bytearray()
INDEX: dict = {}
CHECKS: list[str] = []


def put(name, array):
    a = np.ascontiguousarray(array, dtype=float)
    assert name not in INDEX, name
    INDEX[name] = {"offset": len(BLOB), "count": int(a.size), "shape": list(a.shape), "kind": "f8"}
    BLOB.extend(a.astype("<f8").tobytes())


def check(label, value, bound):
    ok = value <= bound
    CHECKS.append(f"{'ok ' if ok else 'FAIL'} {label}: {value:.3e} <= {bound:.0e}")
    if not ok:
        print("\n".join(CHECKS))
        raise SystemExit(f"self-check failed: {label}")


def case(name, period_atoms, slice_thickness):
    """`period_atoms`: one period of the crystal, snapped, orthogonal; its z length is a whole number of slices."""
    box = tuple(float(x) for x in np.diag(period_atoms.cell))
    period = box[2]
    k = int(round(period / slice_thickness))
    assert abs(k * slice_thickness - period) < 1e-9, (period, slice_thickness)
    gpts = tuple(2 * int(np.ceil(L / SAMPLING / 2)) for L in box[:2])
    sampling = (box[0] / gpts[0], box[1] / gpts[1])
    periods = sorted({max(1, round(t * 100 / period)) for t in TARGET_NM})
    assert len(periods) == 3, periods
    n = max(periods)
    atoms = period_atoms.copy()
    atoms = ase.Atoms(numbers=list(period_atoms.numbers) * n,
                      positions=np.concatenate([period_atoms.positions + [0, 0, m * period] for m in range(n)]),
                      cell=[box[0], box[1], period * n], pbc=True)
    planes = tuple(p * k - 1 for p in periods)
    pot = abtem.Potential(atoms, gpts=gpts, slice_thickness=[slice_thickness] * (k * n), projection="infinite",
                          exit_planes=planes)
    pos = (box[0] * 0.25, box[1] * 0.25)
    exits = abtem.Probe(energy=ENERGY, semiangle_cutoff=CUTOFF).multislice(pot, scan=[pos]).compute()
    waves = np.asarray(exits.array).reshape(len(planes), *gpts)
    pats = np.asarray(abtem.PixelatedDetector(max_angle="cutoff").detect(exits).compute().array)
    pats = pats.reshape(len(planes), *pats.shape[-2:])
    # the text side: the potential slices of one period, tiled
    positions, thick, labels = s1.text_wrap_and_slice(period_atoms.numbers, period_atoms.positions, box, slice_thickness)
    slices = []
    for j in range(len(thick)):
        nums = [int(z) for z, l in zip(period_atoms.numbers, labels) if l == j]
        pj = [p for p, l in zip(positions, labels) if l == j]
        slices.append(s1.text_potential_slice(nums, pj, gpts, sampling, "abTEMFloat32"))
    assert len(slices) == k
    psi = s1.text_probe(ENERGY, gpts, sampling, CUTOFF, {}, pos)
    text = s1.text_multislice(psi, slices * n, list(thick) * n, ENERGY, gpts, sampling, set(planes))
    assert len(text) == len(planes)
    thicknesses = [p * period / 10 for p in periods]  # nm
    for i, (t, w, a) in enumerate(zip(text, waves, pats)):
        check(f"{name} {thicknesses[i]:.1f} nm exit wave text vs abTEM", float(np.abs(t - w).max() / np.abs(w).max()), 1e-10)
        tp = s2.text_pixelated(t, gpts, sampling, "cutoff")
        check(f"{name} {thicknesses[i]:.1f} nm pattern text vs abTEM", float(np.abs(tp - a).max() / np.abs(a).max()), 1e-10)
    put(f"{name}.patterns", pats)
    put(f"{name}.exit_waves_abs", np.abs(waves))
    return {"period_box": list(box), "gpts": list(gpts), "slice_thickness": slice_thickness, "slices_per_period": k,
            "periods": periods, "thickness_nm": thicknesses, "exit_planes": list(planes), "energy": ENERGY,
            "semiangle_cutoff_mrad": CUTOFF, "probe_position": list(pos), "pattern_shape": list(pats.shape[1:]),
            "numbers": [int(z) for z in period_atoms.numbers], "positions": period_atoms.positions.tolist()}


def main():
    s = bi.slab.structures()
    _, u = bi.unit_of(s["Si"], (1, 1, 0))
    si110, k, h = bi.slab_atoms(u, 1)
    al = bulk("Al", "fcc", a=4.05, cubic=True)
    pins = {"generator": "tools/pins/cbed-series/pins.py",
            "versions": {"abtem": abtem.__version__, "ase": ase.__version__, "numpy": np.__version__,
                         "python": platform.python_version()},
            "cases": {}}
    pins["cases"]["Si110"] = case("Si110", si110, h)
    pins["cases"]["Al001"] = case("Al001", al, 2.025)

    os.makedirs(FIXTURES, exist_ok=True)
    blob = bytes(BLOB)
    with open(os.path.join(FIXTURES, "cbed-series-arrays.bin"), "wb") as fh:
        fh.write(blob)
    pins["arrays"] = {"file": "cbed-series-arrays.bin", "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob),
                      "index": INDEX}
    pins["self_checks"] = CHECKS
    with open(os.path.join(FIXTURES, "cbed-series-pins.json"), "w") as fh:
        json.dump(pins, fh, indent=1, sort_keys=True)
    print("\n".join(CHECKS))
    print(f"wrote {len(INDEX)} arrays, {len(blob) / 1e6:.2f} MB; {len(CHECKS)} self-checks passed")


if __name__ == "__main__":
    main()

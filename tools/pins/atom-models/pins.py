"""SimulatoKit specimens from atom models: the pin generator (MK13).

Models built with ASE: Al with a vacancy, a Sigma5 (310) [001] tilt bicrystal in Al, and an orthorhombic cell whose beam
runs along a, b and c (abTEM's `orthogonalize_cell(plane=...)`); and ASE's `Atoms.center(vacuum=)`: on a model with no cell (`all`) and on one that starts from a
6 x 7 x 8 A cell, padded on two axes (`axis_xy`) and on z only (`z_only`).
abTEM 1.0.10 (float64, numpy FFT, eager) gives the orthogonalised atoms and the plane-wave exit wave of each model; the
exit wave is re-derived in NumPy (text_wrap_and_slice, text_potential_slice, text_multislice) and asserted equal to abTEM's
at 1e-10 of the maximum. The padding cases are pure ASE and are written as cell and positions. Writes

  Tests/SimulatoKitParityTests/Fixtures/atom-models-pins.json
  Tests/SimulatoKitParityTests/Fixtures/atom-models-arrays.bin   (float64 little-endian; complex interleaved)

Run from the repo root:  .venv/bin/python -I -B tools/pins/atom-models/pins.py
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

import abtem  # noqa: E402
import ase  # noqa: E402
from abtem.atoms import orthogonalize_cell  # noqa: E402
from ase.build import bulk, molecule  # noqa: E402

ENERGY = 200e3
SLICE = 2.025
SAMPLING = 0.15
A0 = 4.05
BLOB = bytearray()
INDEX: dict = {}
CHECKS: list[str] = []


def put(name, array):
    a = np.ascontiguousarray(array)
    if np.iscomplexobj(a):
        raw, kind = np.stack([a.real, a.imag], axis=-1).astype("<f8").tobytes(), "c16"
    else:
        raw, kind = a.astype("<f8").tobytes(), "f8"
    assert name not in INDEX, name
    INDEX[name] = {"offset": len(BLOB), "count": int(a.size), "shape": list(a.shape), "kind": kind}
    BLOB.extend(raw)


def check(label, value, bound):
    ok = value <= bound
    CHECKS.append(f"{'ok ' if ok else 'FAIL'} {label}: {value:.3e} <= {bound:.0e}")
    if not ok:
        print("\n".join(CHECKS))
        raise SystemExit(f"self-check failed: {label}")


def snap(atoms):
    """Wrap into the cell and snap coordinates at 1e-6 (abTEM 1.0.10's infinite projection drops atoms that wrap to just
    under the cell edge, as tools/pins/slab)."""
    atoms = atoms.copy()
    atoms.wrap()
    atoms.positions = np.round(atoms.positions, 6)
    atoms.positions[np.abs(atoms.positions) < 1e-6] = 0.0
    return atoms


def exit_wave(name, atoms, slice_thickness=SLICE):
    """abTEM's exit wave of an orthogonal model, and the text version, asserted equal; returns the abTEM wave."""
    box = tuple(float(x) for x in np.diag(atoms.cell))
    gpts = tuple(2 * int(np.ceil(L / SAMPLING / 2)) for L in box[:2])
    sampling = (box[0] / gpts[0], box[1] / gpts[1])
    pot = abtem.Potential(atoms, gpts=gpts, slice_thickness=slice_thickness, projection="infinite")
    psi = np.asarray(abtem.PlaneWave(energy=ENERGY).multislice(pot).compute().array)
    positions, thick, labels = s1.text_wrap_and_slice(atoms.numbers, atoms.positions, box, slice_thickness)
    slices = []
    for j in range(len(thick)):
        nums = [int(z) for z, l in zip(atoms.numbers, labels) if l == j]
        pj = [p for p, l in zip(positions, labels) if l == j]
        slices.append(s1.text_potential_slice(nums, pj, gpts, sampling, "abTEMFloat32"))
    text = s1.text_multislice(np.ones(gpts, complex), slices, list(thick), ENERGY, gpts, sampling, {len(thick) - 1})[0]
    check(f"{name} exit wave text vs abTEM", s1.relmax(text, psi), 1e-10)
    put(f"{name}.exit_wave", psi)
    return gpts, box


def model_entry(atoms):
    return {"box": [float(x) for x in np.diag(atoms.cell)], "numbers": [int(z) for z in atoms.numbers],
            "positions": atoms.positions.tolist()}


def vacancy():
    atoms = bulk("Al", "fcc", a=A0, cubic=True) * (3, 3, 4)
    del atoms[len(atoms) // 2]
    return snap(atoms)


def sigma5():
    """A (310) [001] tilt bicrystal, a test specimen and not one ideal boundary (the grains' thickness along x is not a
    multiple of the rotated lattice's period, so the box holds two boundaries with an arbitrary rigid shift): grain A has [3 1 0] along y, grain B [3 -1 0] along y (both repeat every
    sqrt(10) a along y), A fills the left half of the box and B the right half; atoms of the two grains closer than
    1.8 A across either boundary lose the B atom."""
    period = np.sqrt(10.0) * A0
    nz = 2
    half = 3.0 * A0  # each grain's thickness along x
    box = np.array([2 * half, period, nz * A0])
    atoms_all = []
    for sign, x0 in ((1, 0.0), (-1, half)):
        base = bulk("Al", "fcc", a=A0, cubic=True) * (14, 14, nz)
        base.positions[:, :2] -= base.positions[:, :2].mean(0)
        theta = np.arctan(1.0 / 3.0)
        c, s = np.cos(-sign * theta), np.sin(-sign * theta)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        base.positions = base.positions @ rot.T
        # [3 sign 0] now lies along +y; keep the atoms of this grain's slab
        base.positions[:, 0] += x0 + half / 2
        base.positions[:, 1] += period / 2
        keep = (base.positions[:, 0] >= x0 - 1e-9) & (base.positions[:, 0] < x0 + half - 1e-9) & \
               (base.positions[:, 1] >= -1e-9) & (base.positions[:, 1] < period - 1e-9)
        atoms_all.append(base[keep])
    a, b = atoms_all
    ab = a + b
    ab.set_cell(box)
    ab.set_pbc(True)
    from ase.neighborlist import neighbor_list
    n_a = len(a)
    i, j, d = neighbor_list("ijd", ab, 1.8)
    drop = sorted({int(max(p, q)) for p, q in zip(i, j) if (p < n_a) != (q < n_a)}, reverse=True)
    for k in drop:
        del ab[k]
    out = snap(ab)
    from ase.geometry import get_distances
    _, dist = get_distances(out.positions, cell=out.cell, pbc=True)
    np.fill_diagonal(dist, 9.9)
    assert dist.min() > 1.7, dist.min()  # no overlapping atoms, wrap included
    return out


def main():
    pins = {"generator": "tools/pins/atom-models/pins.py",
            "versions": {"abtem": abtem.__version__, "ase": ase.__version__, "numpy": np.__version__,
                         "python": platform.python_version()},
            "energy": ENERGY, "slice_thickness": SLICE, "cases": {}}
    v = vacancy()
    gpts, box = exit_wave("vacancy", v)
    pins["cases"]["vacancy"] = {**model_entry(v), "gpts": list(gpts), "full_atoms_before_vacancy": len(v) + 1}
    s5 = sigma5()
    gpts, box = exit_wave("sigma5", s5)
    pins["cases"]["sigma5"] = {**model_entry(s5), "gpts": list(gpts)}

    # The beam along a, b and c of an orthorhombic cell: abTEM's plane strings, then the exit wave of each.
    ortho = bulk("Al", "fcc", a=A0, cubic=True) * (2, 3, 4)
    ortho.positions = np.round(ortho.positions, 6)
    pins["cases"]["ortho"] = {"box": [float(x) for x in np.diag(ortho.cell)], "beams": {}}
    for beam, plane in (("c", "xy"), ("a", "yz"), ("b", "zx")):
        out = orthogonalize_cell(ortho.copy(), plane=plane, max_repetitions=1)
        out = snap(out)
        gpts, box = exit_wave(f"ortho_{beam}", out)
        pins["cases"]["ortho"]["beams"][beam] = {**model_entry(out), "plane": plane, "gpts": list(gpts)}

    # ASE's center(vacuum=) on a model with no cell, and on one axis only.
    pad = {}
    for name, kwargs in (("all", {"vacuum": 3.0}), ("axis_xy", {"vacuum": 2.5, "axis": (0, 1)}),
                         ("z_only", {"vacuum": 4.0, "axis": 2})):
        atoms = molecule("CH3CH2OH")
        if name != "all":
            atoms.set_cell(np.diag([6.0, 7.0, 8.0]))
        atoms.center(**kwargs)
        pad[name] = {"cell": np.array(atoms.cell).tolist(), "numbers": [int(z) for z in atoms.numbers],
                     "positions": atoms.positions.tolist(), "arguments": {k: (list(x) if isinstance(x, tuple) else x)
                                                                           for k, x in kwargs.items()}}
    pins["cases"]["padding"] = pad

    os.makedirs(FIXTURES, exist_ok=True)
    blob = bytes(BLOB)
    with open(os.path.join(FIXTURES, "atom-models-arrays.bin"), "wb") as fh:
        fh.write(blob)
    pins["arrays"] = {"file": "atom-models-arrays.bin", "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob),
                      "index": INDEX}
    pins["self_checks"] = CHECKS
    with open(os.path.join(FIXTURES, "atom-models-pins.json"), "w") as fh:
        json.dump(pins, fh, indent=1, sort_keys=True)
    print("\n".join(CHECKS))
    print(f"wrote {len(INDEX)} arrays, {len(blob) / 1e6:.2f} MB; {len(CHECKS)} self-checks passed")


if __name__ == "__main__":
    main()

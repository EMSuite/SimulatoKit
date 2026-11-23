"""SimulatoKit slab builder: a crystal cell turned to a zone axis and made orthogonal (abTEM 1.0.10 `orthogonalize_cell`).

Writes Tests/SimulatoKitParityTests/Fixtures/slab-pins.json and slab-arrays.bin:

* units: the orthogonal repeat unit of each named structure and zone axis, at abTEM's default of 5 repetitions and at the
  repetitions the kit's ladder (5, 10, 20, 40) picks for a strain limit of 0.1 %: the box, the integer supercell matrix,
  the atoms (Z and Cartesian position), and the scale and shear of the affine transform abTEM applies
  (`decompose_affine_transform`). The tests hold the box, the scale and the shear at rtol 1e-7 (assert_allclose's default
  for float64; abTEM states no tolerance), the matrix and the atom counts exact, and the atoms as periodic sets at
  rtol 1e-7 with atol 1e-9 A.
* ladder: what the kit's ladder returns for each structure: the repetitions it stops at, or the refusal with the least
  strain found.
* orientation: plain matrices of DiffractoKit's `crystal-orientation-pins.json` (py4DSTEM f050d207 `parse_orientation`,
  in-plane angle 0) that the slab's rotation is checked against: the same third column (the zone axis, pointing at the
  source) and a turn about the beam only (the kit puts a lattice row along x, py4DSTEM a fixed reference).
* slab: Si [110], 200 kV, one orthogonal unit laterally and 10 periods deep, cut in slices of at most 1 A that repeat with
  the period (8 per period): the built slab's atoms (after the kit's z shift, which keeps every atomic plane off a slice
  boundary) and abTEM's exit wave of exactly those atoms and slices (`PlaneWave.multislice`, infinite projection, float64). The test holds it at the kit's multislice pin (1e-10 of
  the largest value with abTEM's float32 pi, 1e-6 with the exact pi).

The rotation that turns the cell to the zone axis is the kit's (SlabBuilder.swift), restated here: z along the zone axis,
x along the shortest lattice row at right angles to it (the one closest to py4DSTEM's x reference on a tie), y = z x x.
abTEM's `orthogonalize_cell(plane=(x, y))` with two vectors fails in 1.0.10 (UnboundLocalError in
`rotate_atoms_to_plane`), so the atoms are turned with NumPy first and `orthogonalize_cell` runs with the default plane.

abTEM's cut (ASE's `cut`) wraps fractional coordinates with `% 1.0`; a coordinate a rounding error below zero becomes one
just below 1, lands on the far face of the new cell and is dropped (Al and Cu [111] lose 1 of 24 atoms). Every reference
unit must therefore hold det(supercell) times the cell's atoms; where the plain call falls short, the generator calls
`orthogonalize_cell` again with `origin` moved off the atoms (ORIGIN below) and shifts the result back, and the case
records `abtem_origin`. The kit takes the new cell's atoms by exact lattice arithmetic and needs no such step.

Run from the repo root:  .venv/bin/python -I -B tools/pins/slab/pins.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform

import numpy as np
import abtem
import ase
from ase.build import bulk
from ase.spacegroup import crystal
from abtem.atoms import best_orthogonal_cell, decompose_affine_transform, orthogonalize_cell

assert abtem.__version__ == "1.0.10", abtem.__version__
assert ase.__version__ == "3.29.0", ase.__version__
abtem.config.set({"precision": "float64", "fft": "numpy", "dask.lazy": False, "diagnostics.progress_bar": False})

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "..", "Tests", "SimulatoKitParityTests", "Fixtures")
BLOB = bytearray()
INDEX: dict = {}

ORIGIN = np.array([0.0123, 0.0234, 0.0345])  # A, in the turned frame; used only where the plain call drops atoms
LADDER = (5, 10, 20, 40)
MAXIMUM_STRAIN = 0.001
ROW_SEARCH = 10  # the lattice rows searched for the x axis: |i|, |j|, |k| <= 10
SHIFT_STEPS = 720  # the z shifts tried: j * slice height / 720


def put(name, a):
    a = np.ascontiguousarray(a)
    raw = np.stack([a.real, a.imag], axis=-1).astype("<f8").tobytes()
    INDEX[name] = {"offset": len(BLOB), "count": int(a.size), "shape": list(a.shape), "kind": "c16"}
    BLOB.extend(raw)


def check(label, value, bound):
    if not value <= bound:
        raise SystemExit(f"FAILED {label}: {value} > {bound}")
    print(f"ok  {label}: {value:.3e} <= {bound:.0e}")


# --------------------------------------------------------------------------------------------------------------------
# The structures (public lattice parameters; beta'' is Andersen et al., Acta Mater. 46 (1998) 3283, as DiffractoKit's
# fixture writes it).
# --------------------------------------------------------------------------------------------------------------------

def structures():
    s = {}
    s["Al"] = bulk("Al", "fcc", a=4.05, cubic=True)
    s["Si"] = bulk("Si", "diamond", a=5.431, cubic=True)
    s["Cu"] = bulk("Cu", "fcc", a=3.615, cubic=True)
    s["Fe"] = bulk("Fe", "bcc", a=2.8665, cubic=True)
    s["Mg"] = crystal("Mg", [(1 / 3, 2 / 3, 0.25)], spacegroup=194, cellpar=[3.209, 3.209, 5.211, 90, 90, 120])
    s["SrTiO3"] = crystal(["Sr", "Ti", "O"], [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)], spacegroup=221,
                          cellpar=[3.905, 3.905, 3.905, 90, 90, 90])
    s["beta2"] = crystal(["Mg", "Mg", "Mg", "Si", "Si", "Si"],
                         [(0, 0, 0), (0.3459, 0, 0.089), (0.430, 0, 0.652), (0.0565, 0, 0.649), (0.1885, 0, 0.224),
                          (0.2171, 0, 0.617)], spacegroup=12, cellpar=[15.16, 4.05, 6.74, 90, 105.3, 90])
    return s


# Mg [2 -1 -1 0] in four indices is [1 0 0] in three.
CASES = [("Al", (0, 0, 1)), ("Al", (1, 1, 0)), ("Al", (1, 1, 1)), ("Al", (1, 1, 2)),
         ("Si", (0, 0, 1)), ("Si", (1, 1, 0)), ("Si", (1, 1, 1)), ("Si", (1, 1, 2)),
         ("Cu", (0, 0, 1)), ("Cu", (1, 1, 0)), ("Cu", (1, 1, 1)), ("Cu", (1, 1, 2)),
         ("Fe", (0, 0, 1)), ("Fe", (1, 1, 1)),
         ("Mg", (0, 0, 1)), ("Mg", (1, 0, 0)), ("Mg", (1, 0, 1)),
         ("SrTiO3", (0, 0, 1)), ("SrTiO3", (1, 1, 0)),
         ("beta2", (0, 1, 0)), ("beta2", (1, 0, 0))]
REFUSED = [("Mg", (1, 0, 3))]  # 0.22 % at every rung: the ladder refuses it at 0.1 %

# py4DSTEM's plain matrices on its own cells, copied from DiffractoKit's Tests/DiffractoCoreTests/Fixtures/
# crystal-orientation-pins.json (py4DSTEM f050d207 `parse_orientation`, in-plane angle 0): `cell` rows are a, b, c in A;
# `matrix` is row-major, its columns lab x, y and z in the crystal's Cartesian frame.
ORIENTATION = [
    {"phase": "Al", "cell": [[4.0495, 0.0, 0.0], [2.4796036065736033e-16, 4.0495, 0.0], [2.4796036065736033e-16, 2.4796036065736033e-16, 4.0495]], "zone": [0, 0, 1], "matrix": [1.0, -3.749399456654644e-33, 6.123233995736766e-17, 0.0, 1.0, 6.123233995736766e-17, -6.123233995736766e-17, -6.123233995736766e-17, 1.0]},
    {"phase": "Al", "cell": [[4.0495, 0.0, 0.0], [2.4796036065736033e-16, 4.0495, 0.0], [2.4796036065736033e-16, 2.4796036065736033e-16, 4.0495]], "zone": [0, 1, 1], "matrix": [8.659560562354932e-17, -1.0, 8.659560562354933e-17, 0.7071067811865476, 1.2246467991473532e-16, 0.7071067811865476, -0.7071067811865476, 0.0, 0.7071067811865476]},
    {"phase": "Al", "cell": [[4.0495, 0.0, 0.0], [2.4796036065736033e-16, 4.0495, 0.0], [2.4796036065736033e-16, 2.4796036065736033e-16, 4.0495]], "zone": [1, 1, 1], "matrix": [0.408248290463863, -0.7071067811865475, 0.5773502691896258, 0.408248290463863, 0.7071067811865475, 0.5773502691896258, -0.816496580927726, 0.0, 0.5773502691896258]},
    {"phase": "Al", "cell": [[4.0495, 0.0, 0.0], [2.4796036065736033e-16, 4.0495, 0.0], [2.4796036065736033e-16, 2.4796036065736033e-16, 4.0495]], "zone": [1, 1, 2], "matrix": [0.5773502691896256, -0.7071067811865476, 0.40824829046386313, 0.5773502691896256, 0.7071067811865476, 0.40824829046386313, -0.5773502691896258, 0.0, 0.816496580927726]},
    {"phase": "Al", "cell": [[4.0495, 0.0, 0.0], [2.4796036065736033e-16, 4.0495, 0.0], [2.4796036065736033e-16, 2.4796036065736033e-16, 4.0495]], "zone": [1, 2, 3], "matrix": [0.3585685828003182, -0.8944271909999159, 0.2672612419124245, 0.7171371656006361, 0.4472135954999581, 0.5345224838248488, -0.5976143046671969, 0.0, 0.8017837257372732]},
    {"phase": "beta_double_prime_Mg5Si6", "cell": [[15.16, 0.0, 0.0], [2.4799097682733903e-16, 4.05, 0.0], [-1.7785043567666132, 5.216079547018508e-16, 6.501117000405559]], "zone": [0, 0, 1], "matrix": [-0.9645574184577981, -2.932846089003422e-16, -0.26387304996537286, 2.82889845234319e-16, -1.0, 7.738990425843483e-17, -0.26387304996537286, 0.0, 0.9645574184577981]},
    {"phase": "beta_double_prime_Mg5Si6", "cell": [[15.16, 0.0, 0.0], [2.4799097682733903e-16, 4.05, 0.0], [-1.7785043567666132, 5.216079547018508e-16, 6.501117000405559]], "zone": [1, 1, 1], "matrix": [0.4035642437040253, -0.2896799134486618, 0.8678833152837852, 0.12214144297298984, 0.9571235801840721, 0.2626707447815612, -0.9067620245202563, 0.0, 0.42164277639718445]},
    {"phase": "beta_double_prime_Mg5Si6", "cell": [[15.16, 0.0, 0.0], [2.4799097682733903e-16, 4.05, 0.0], [-1.7785043567666132, 5.216079547018508e-16, 6.501117000405559]], "zone": [0, 1, 0], "matrix": [0.0, -1.0, 6.123233995736766e-17, 0.0, 6.123233995736766e-17, 1.0, -1.0, 0.0, 0.0]},
    {"phase": "Mg", "cell": [[3.2094, 0.0, 0.0], [-1.6046999999999993, 2.7794219309057775, 0.0], [3.190694770498514e-16, 5.526445453947743e-16, 5.210800000000001]], "zone": [0, 0, 1], "matrix": [1.0, -6.494150356796982e-33, 6.123233995736765e-17, 0.0, 1.0, 1.0605752387249064e-16, -6.123233995736765e-17, -1.0605752387249064e-16, 1.0]},
    {"phase": "Mg", "cell": [[3.2094, 0.0, 0.0], [-1.6046999999999993, 2.7794219309057775, 0.0], [3.190694770498514e-16, 5.526445453947743e-16, 5.210800000000001]], "zone": [1, 0, 0], "matrix": [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0]},
    {"phase": "Mg", "cell": [[3.2094, 0.0, 0.0], [-1.6046999999999993, 2.7794219309057775, 0.0], [3.190694770498514e-16, 5.526445453947743e-16, 5.210800000000001]], "zone": [1, 1, 0], "matrix": [0.0, -0.8660254037844386, 0.5000000000000001, 0.0, 0.5000000000000001, 0.8660254037844386, -1.0, 0.0, 0.0]},
    {"phase": "Mg", "cell": [[3.2094, 0.0, 0.0], [-1.6046999999999993, 2.7794219309057775, 0.0], [3.190694770498514e-16, 5.526445453947743e-16, 5.210800000000001]], "zone": [2, 1, 0], "matrix": [0.0, -0.5, 0.8660254037844387, 0.0, 0.8660254037844387, 0.5, -1.0, 0.0, 0.0]},
    {"phase": "Mg", "cell": [[3.2094, 0.0, 0.0], [-1.6046999999999993, 2.7794219309057775, 0.0], [3.190694770498514e-16, 5.526445453947743e-16, 5.210800000000001]], "zone": [1, 1, 1], "matrix": [0.425728701208412, -0.8660254037844385, 0.26221188563335335, 0.7373837407332791, 0.5000000000000001, 0.4541643082654076, -0.5244237712667066, 0.0, 0.8514574024168238]},
]


# --------------------------------------------------------------------------------------------------------------------
# The kit's rotation, restated (SlabBuilder.zoneRotation).
# --------------------------------------------------------------------------------------------------------------------

def py4dstem_x(ez):
    x = np.cross([0.0, 1.0, 0.0], ez) if abs(ez[2]) > 1 - 1e-6 else np.array([0.0, 0.0, -1.0])
    y = np.cross(ez, x)
    x = np.cross(y, ez)
    return x / np.linalg.norm(x)


def zone_rotation(cell, uvw):
    """Rows ex, ey, ez: p_turned = R p_crystal."""
    cell = np.asarray(cell, dtype=float)
    d = np.asarray(uvw, dtype=float) @ cell
    ez = d / np.linalg.norm(d)
    xref = py4dstem_x(ez)
    best = None
    n = ROW_SEARCH
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            for k in range(-n, n + 1):
                if i == 0 and j == 0 and k == 0:
                    continue
                r = np.array([i, j, k], dtype=float) @ cell
                length = math.sqrt(float(r @ r))
                if abs(float(r @ ez)) > 1e-9 * length:
                    continue
                c = float(r @ xref) / length
                if best is None:
                    best = (length, c, (i, j, k), r)
                    continue
                bl, bc, bn, _ = best
                shorter = length < bl * (1 - 1e-9)
                tie = length <= bl * (1 + 1e-9)
                if shorter or (tie and (c > bc + 1e-9 or (c >= bc - 1e-9 and (i, j, k) > bn))):
                    best = (length, c, (i, j, k), r)
    ex = xref if best is None else best[3] / best[0]
    ey = np.cross(ez, ex)
    ey = ey / np.linalg.norm(ey)
    return np.array([ex, ey, ez]), (None if best is None else list(best[2]))


# --------------------------------------------------------------------------------------------------------------------
# The reference: abTEM's orthogonalize_cell on the turned atoms.
# --------------------------------------------------------------------------------------------------------------------

def reference_unit(atoms, R, repetitions):
    turned = atoms.copy()
    turned.positions[:] = turned.positions @ R.T
    turned.cell[:] = np.array(turned.cell) @ R.T
    zeroed = np.array(turned.cell)
    zeroed[np.abs(zeroed) < 1e-6] = 0.0  # orthogonalize_cell's own first step, for the supercell matrix below
    box = best_orthogonal_cell(zeroed, max_repetitions=repetitions)
    if tuple(np.diag(zeroed)) == tuple(box) and not np.any(np.abs(zeroed[~np.eye(3, dtype=bool)]) > 1e-12):
        supercell = np.eye(3, dtype=int)
    else:
        supercell = np.round(np.dot(np.diag(box), np.linalg.inv(zeroed))).astype(int)
    expected = round(abs(np.linalg.det(supercell))) * len(atoms)
    for origin in (np.zeros(3), ORIGIN):
        out = orthogonalize_cell(turned.copy(), max_repetitions=repetitions, return_transform_matrix=True,
                                 origin=tuple(float(v) for v in origin))
        unit, A = out if isinstance(out, tuple) else (out, np.eye(3))  # the early return drops the matrix
        if len(unit) == expected:
            break
    else:
        raise SystemExit(f"abTEM keeps {len(unit)} of {expected} atoms with and without the origin shift")
    lengths = np.diag(np.array(unit.cell))
    check("unit box equals best_orthogonal_cell", float(np.max(np.abs(lengths - box))), 0.0)
    positions = (unit.positions + origin @ A) % lengths
    _, scale, shear = decompose_affine_transform(A)
    return {
        "box": [float(v) for v in lengths],
        "supercell": supercell.tolist(),
        "numbers": [int(z) for z in unit.numbers],
        "positions": positions.tolist(),
        "scale": [float(v) for v in scale],
        "shear": [float(v) for v in shear],
        "strain": float(max(np.max(np.abs(np.abs(scale) - 1)), np.max(np.abs(shear)))),
        "abtem_origin": [float(v) for v in origin],
    }


def ladder(atoms, R):
    best = None
    for n in LADDER:
        u = reference_unit(atoms, R, n)
        if u["strain"] <= MAXIMUM_STRAIN:
            return n, u, None
        if best is None or u["strain"] < best[1]["strain"]:
            best = (n, u)
    return None, None, best


# --------------------------------------------------------------------------------------------------------------------
# The slab: the kit's z shift, restated (SlabBuilder.slab), and abTEM's exit wave of the result.
# --------------------------------------------------------------------------------------------------------------------

def slice_plan(period, slice_thickness):
    """The kit's slices of one period: k = ceil(period / s) equal slices, so the boundaries repeat with the crystal."""
    k = math.ceil(period / slice_thickness)
    return k, period / k


def z_shift(z, slice_height):
    """The kit's shift along z: of j * h / 720 (j = 0 ... 719, h the slice height) the one that keeps the atomic planes
    farthest from a slice boundary (distances rounded to 1e-9 A; the smallest j on a tie)."""
    planes = list(z)
    best_j, best_f = 0, -1.0
    for j in range(SHIFT_STEPS):
        delta = j * slice_height / SHIFT_STEPS
        f = math.inf
        for p in planes:
            q = (p + delta) / slice_height
            f = min(f, abs(q - math.floor(q + 0.5)) * slice_height)
        f = round(f * 1e9) / 1e9
        if f > best_f:
            best_j, best_f = j, f
    return best_j * slice_height / SHIFT_STEPS, best_f


def main():
    s = structures()
    pins = {"generator": "tools/pins/slab/pins.py",
            "versions": {"abtem": abtem.__version__, "ase": ase.__version__, "numpy": np.__version__,
                         "python": platform.python_version()},
            "maximum_strain": MAXIMUM_STRAIN, "ladder": list(LADDER), "origin": ORIGIN.tolist()}
    pins["structures"] = {}
    for name, atoms in s.items():
        pins["structures"][name] = {
            "cell": np.array(atoms.cell).tolist(),
            "sites": [[int(z)] + [float(v) for v in f] for z, f in zip(atoms.numbers, atoms.get_scaled_positions())]}

    units, ladders = [], []
    for name, uvw in CASES + REFUSED:
        atoms = s[name]
        R, row = zone_rotation(np.array(atoms.cell), uvw)
        stop, chosen, refusal = ladder(atoms, R)
        entry = {"structure": name, "zone": list(uvw), "x_row": row,
                 "orientation": R.T.tolist()}  # columns: box x, y, z in the crystal frame (the suite's convention)
        if stop is not None:
            entry["repetitions"] = stop
            entry["box"] = chosen["box"]
        else:
            entry["refused"] = {"repetitions": refusal[0], "strain": refusal[1]["strain"]}
        ladders.append(entry)
        if (name, uvw) in REFUSED:
            continue
        for n in sorted({5} | ({stop} if stop else set())):
            u = reference_unit(atoms, R, n)
            u.update({"structure": name, "zone": list(uvw), "repetitions": n})
            units.append(u)
            print(f"{name} {uvw} n={n}: box {np.round(u['box'], 4).tolist()} atoms {len(u['numbers'])} "
                  f"strain {u['strain'] * 100:.4f} %")
    pins["units"] = units
    pins["ladders"] = ladders

    pins["orientation"] = ORIENTATION

    # The Si [110] slab and its exit wave.
    si = s["Si"]
    R, _ = zone_rotation(np.array(si.cell), (1, 1, 0))
    u = reference_unit(si, R, 5)
    period, periods, slice_thickness = u["box"][2], 10, 1.0
    z = [p[2] for p in u["positions"]]
    k, h = slice_plan(period, slice_thickness)
    delta, margin = z_shift(z, h)
    unit_pos = np.array(u["positions"])
    unit_pos[:, 2] = (unit_pos[:, 2] + delta) % period
    numbers = u["numbers"] * periods
    positions = np.concatenate([unit_pos + [0, 0, m * period] for m in range(periods)])
    box = [u["box"][0], u["box"][1], period * periods]
    atoms = ase.Atoms(numbers=numbers, positions=positions, cell=box, pbc=True)
    gpts, energy = (64, 90), 200e3
    thicknesses = [h] * (k * periods)
    pot = abtem.Potential(atoms, gpts=gpts, slice_thickness=thicknesses, projection="infinite")
    exit_wave = np.asarray(abtem.PlaneWave(energy=energy).multislice(pot).compute().array)
    check("Si [110] slab keeps every atom off a slice boundary (A)", -margin, -0.4)
    put("si110.exit", exit_wave)
    pins["slab"] = {"structure": "Si", "zone": [1, 1, 0], "repetitions": 5, "periods": periods,
                    "slice_thickness": slice_thickness, "slices": len(pot.slice_thickness), "slices_per_period": k, "slice_height": h, "z_shift": delta,
                    "plane_margin": margin, "box": box, "gpts": list(gpts), "energy": energy,
                    "numbers": [int(v) for v in numbers], "positions": positions.tolist(),
                    "exit_max_abs": float(np.max(np.abs(exit_wave)))}

    blob = bytes(BLOB)
    pins["arrays"] = {"file": "slab-arrays.bin", "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob),
                      "index": INDEX}
    with open(os.path.join(OUT, "slab-arrays.bin"), "wb") as fh:
        fh.write(blob)
    with open(os.path.join(OUT, "slab-pins.json"), "w") as fh:
        json.dump(pins, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print("slices", len(pot.slice_thickness), "z shift", delta, "margin", margin, "exit max", pins["slab"]["exit_max_abs"])


if __name__ == "__main__":
    main()

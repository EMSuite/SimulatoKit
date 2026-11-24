"""SimulatoKit thermal vibration amplitude per site, with the B and sigma conversions: the pin generator.

abTEM 1.0.10 `FrozenPhonons` (inelastic/phonons.py) with sigma given per element (a dict) and per site (one value per
atom, a list), float64, NumPy FFT, eager. abTEM displaces atom i along each axis in `directions` by sigma_i * r[i, axis],
with r = numpy.random.default_rng(seed).normal(size=(atoms, 3)) per configuration: sigma is the standard deviation per
axis (the isotropic case of `validate_sigmas`). The generator re-derives the displacements from that formula
(`text_displacements`), asserts that abTEM's displaced positions equal positions + text displacements to the last bit
and that abTEM's potential of every configuration equals the published formula (tools/pins/potential-multislice), then
writes the text displacements (which therefore reproduce abTEM's displaced positions exactly), the sigma of every atom
and every slice of abTEM's potential of every configuration.

The conversions: abTEM's `sigma_to_B` and `B_to_sigma` (atoms.py), B = 8 pi^2 sigma^2 with sigma per axis, so B is the
crystallographic isotropic temperature factor (B = 8 pi^2 <u_x^2>). Written as pairs over a ladder of values.

Cases: Si (diamond, 2 x 2 x 1 cubic cells) and SrTiO3 (2 x 2 x 2 cubic cells), each with sigma per element and per
site, two configurations each.

  Tests/SimulatoKitParityTests/Fixtures/thermal-sites-pins.json
  Tests/SimulatoKitParityTests/Fixtures/thermal-sites-arrays.bin   (float64 little-endian)

Run from the repo root:  .venv/bin/python -I -B tools/pins/thermal-sites/pins.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIXTURES = os.path.join(ROOT, "Tests", "SimulatoKitParityTests", "Fixtures")

spec = importlib.util.spec_from_file_location("s1", os.path.join(ROOT, "tools", "pins", "potential-multislice", "pins.py"))
s1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s1)  # sets abTEM's configuration (float64, numpy FFT, eager)

import abtem  # noqa: E402
from abtem.atoms import B_to_sigma, sigma_to_B, validate_sigmas  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.build import bulk  # noqa: E402

assert abtem.__version__ == "1.0.10", abtem.__version__

SYMBOLS = s1.SYMBOLS
BLOB = bytearray()
INDEX: dict = {}
CHECKS: list[str] = []
relmax = s1.relmax


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


def text_site_sigmas(numbers, sigmas):
    """sigma of every atom: the per-element dict resolved by atomic number, or the per-site list as given."""
    if isinstance(sigmas, dict):
        return np.array([float(sigmas[SYMBOLS[z]]) for z in numbers])
    return np.array([float(s) for s in sigmas])


def text_displacements(site_sigmas, seed, directions="xyz"):
    r = np.random.default_rng(seed).normal(size=(len(site_sigmas), 3))
    d = np.zeros((len(site_sigmas), 3))
    for axis, name in enumerate("xyz"):
        if name in directions:
            d[:, axis] = site_sigmas * r[:, axis]
    return d


def text_potential(atoms, d, gpts, slice_thickness):
    box = tuple(float(x) for x in np.diag(atoms.cell))
    sampling = (box[0] / gpts[0], box[1] / gpts[1])
    pos, thick, labels = s1.text_wrap_and_slice(atoms.numbers, atoms.positions + d, box, slice_thickness)
    slices = []
    for j in range(len(thick)):
        nums = [int(z) for z, l in zip(atoms.numbers, labels) if l == j]
        pj = [p for p, l in zip(pos, labels) if l == j]
        slices.append(s1.text_potential_slice(nums, pj, gpts, sampling, "abTEMFloat32"))
    return slices, thick


def case(name, atoms, sigmas, gpts, slice_thickness, num_configs, seed):
    fp = abtem.FrozenPhonons(atoms, num_configs=num_configs, sigmas=sigmas, seed=seed)
    seeds = [int(s) for s in fp.seed]
    site = text_site_sigmas(atoms.numbers, sigmas)
    resolved = validate_sigmas(atoms, sigmas, return_array=True)[0]
    check(f"{name} sigma per atom: text vs abTEM", float(np.max(np.abs(np.asarray(resolved) - site))), 0.0)
    ensemble = fp.to_atoms_ensemble().trajectory
    pot = abtem.Potential(fp, gpts=gpts, slice_thickness=slice_thickness)
    built = np.asarray(pot.build().compute().array)
    disp = []
    for c, s in enumerate(seeds):
        d = text_displacements(site, s)
        moved = np.asarray(ensemble[c].positions)
        check(f"{name} config {c} displaced positions: abTEM vs positions + sigma r (exact)",
              float(np.max(np.abs(moved - (atoms.positions + d)))), 0.0)
        slices, thick = text_potential(atoms, d, gpts, slice_thickness)
        for j, v in enumerate(slices):
            check(f"{name} config {c} V slice {j} text vs abTEM", relmax(v, built[c, j]), 1e-12)
        disp.append(d)
        put(f"{name}.displacement.{c}", d)
        for j in range(built.shape[1]):
            put(f"{name}.V.config{c}.slice{j}", built[c, j])
    put(f"{name}.site_sigmas", site)
    return {"box": [float(x) for x in np.diag(atoms.cell)], "gpts": list(gpts), "slice_thickness_requested": slice_thickness,
            "numbers": [int(z) for z in atoms.numbers], "positions": atoms.positions.tolist(),
            "sigmas": sigmas if isinstance(sigmas, dict) else None, "per_site": not isinstance(sigmas, dict),
            "num_configs": num_configs, "seed": seed, "seeds": seeds, "slices": int(built.shape[1]),
            "slice_thickness": list(text_potential(atoms, disp[0], gpts, slice_thickness)[1])}


def main():
    pins = {"versions": {"abtem": abtem.__version__, "numpy": np.__version__, "python": sys.version.split()[0]},
            "cases": {}}
    si = bulk("Si", "diamond", a=5.431, cubic=True) * (2, 2, 1)
    # Per site: eight different sigmas, repeated over the four cells (a different sigma on every site of the cell).
    si_sites = [0.05 + 0.005 * (i % 8) for i in range(len(si))]
    a0 = 3.905
    sto = Atoms("SrTiO3", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
                cell=[a0, a0, a0], pbc=True) * (2, 2, 2)
    # Per site: Sr 0.08, Ti 0.06, the three O sites of the cell 0.06, 0.07, 0.09 (as an anisotropic environment would
    # give them), and the first cell's Sr raised to 0.10, so no element has one sigma.
    base = {"Sr": 0.08, "Ti": 0.06}
    o_ladder = [0.06, 0.07, 0.09]
    sto_sites = []
    for i, z in enumerate(sto.numbers):
        k = i % 5
        sym = SYMBOLS[z]
        sto_sites.append(o_ladder[k - 2] if sym == "O" else base[sym])
    sto_sites[0] = 0.10
    pins["cases"]["T1"] = case("T1", si, {"Si": 0.075}, (64, 64), 5.431 / 4, 2, 11)
    pins["cases"]["T2"] = case("T2", si, si_sites, (64, 64), 5.431 / 4, 2, 11)
    pins["cases"]["T3"] = case("T3", sto, {"Sr": 0.08, "Ti": 0.06, "O": 0.07}, (64, 64), a0 / 2, 2, 13)
    pins["cases"]["T4"] = case("T4", sto, sto_sites, (64, 64), a0 / 2, 2, 13)
    # The per-site path with every site at its element's sigma is the per-element path, to the last bit.
    same = text_site_sigmas(sto.numbers, {"Sr": 0.08, "Ti": 0.06, "O": 0.07}).tolist()
    a = abtem.FrozenPhonons(sto, num_configs=2, sigmas=same, seed=13).to_atoms_ensemble().trajectory
    b = abtem.FrozenPhonons(sto, num_configs=2, sigmas={"Sr": 0.08, "Ti": 0.06, "O": 0.07}, seed=13).to_atoms_ensemble().trajectory
    check("per site at the element sigmas vs per element (abTEM, exact)",
          float(max(np.max(np.abs(x.positions - y.positions)) for x, y in zip(a, b))), 0.0)

    sigmas = np.array([0.0, 1e-4, 0.01, 0.05, 0.0707, 0.075, 0.08, 0.1, 0.15, 0.3, 1.0])
    bs = np.array([0.0, 1e-6, 0.01, 0.1, 0.25, 0.4, 0.5, 0.79, 1.0, 2.0, 10.0])
    b_of_sigma = np.asarray(sigma_to_B(sigmas), dtype=float)
    sigma_of_b = np.asarray(B_to_sigma(bs), dtype=float)
    check("sigma_to_B vs 8 pi^2 sigma^2", relmax(b_of_sigma, 8 * np.pi ** 2 * sigmas ** 2), 1e-15)
    check("B_to_sigma vs sqrt(B / (8 pi^2))", relmax(sigma_of_b, np.sqrt(bs / (8 * np.pi ** 2))), 1e-15)
    pins["conversions"] = {"sigma": sigmas.tolist(), "sigma_to_B": b_of_sigma.tolist(), "B": bs.tolist(),
                           "B_to_sigma": sigma_of_b.tolist()}

    os.makedirs(FIXTURES, exist_ok=True)
    blob = bytes(BLOB)
    with open(os.path.join(FIXTURES, "thermal-sites-arrays.bin"), "wb") as fh:
        fh.write(blob)
    pins["arrays"] = {"file": "thermal-sites-arrays.bin", "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob),
                      "index": INDEX}
    pins["self_checks"] = CHECKS
    with open(os.path.join(FIXTURES, "thermal-sites-pins.json"), "w") as fh:
        json.dump(pins, fh, indent=1, sort_keys=True)
    print("\n".join(CHECKS))
    print(f"wrote {len(INDEX)} arrays, {len(blob) / 1e6:.2f} MB; {len(CHECKS)} self-checks passed")


if __name__ == "__main__":
    main()

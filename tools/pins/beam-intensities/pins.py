"""SimulatoKit beam intensities by hkl: the dynamical intensity of each beam of a slab's exit waves, indexed by the
crystal's Miller indices, at every stored thickness (abTEM 1.0.10 `DiffractionPatterns.index_diffraction_spots`).

Writes Tests/SimulatoKitParityTests/Fixtures/beam-intensities-pins.json:

* cases: Si [110], Si [111], Mg [0001] and Si [001] at 200 kV, the kit's slab (the orthogonal unit of `tools/pins/slab/pins.py`,
  one unit laterally, slices of at most 1 A that repeat with the period, the kit's z shift and mirror to depth), stored after five whole
  numbers of periods from about 2 to 100 nm. For each case: the grid, the periods stored, the exit planes, and for every
  zero-order Laue zone beam inside the antialias aperture its crystal hkl and its intensity at each stored thickness.

The reference is abTEM's own indexing: `PlaneWave.multislice` (infinite projection, float64) with the stored exit planes,
`diffraction_patterns()` (cropped at the antialias cutoff, the zero frequency centred) and `index_diffraction_spots` with
the crystal cell as it sits in the box (the strained cell inv(V) diag(box), V the integer supercell matrix), the lattice's
centring (`centering`: F for Si, P for Mg; abTEM then leaves out every hkl its `get_reflection_condition` forbids), no
orientation matrix and no radius: abTEM 1.0.10 then reads the single pixel the reflection falls in (`np.digitize` on the
pixel edges), not a disc, and zeroes a spot whose pixel a reflection of smaller excitation error already holds. In that
cell's reciprocal lattice the reflection hkl falls exactly on the pixel (V hkl)_xy of the box, which is the kit's integer
map. The generator checks that abTEM gives every pixel the kit's rule selects (V hkl has z = 0, a reflection of the
centred lattice, inside the cropped pattern, |g| at most its largest frequency, abTEM's default g_max) to the same hkl, and
that abTEM's value is that pixel's. Si [001] is the case of SimulatoKit#145: its box is the conventional cubic cell, so
half its in-zone integer hkl (1 0 0, 1 2 0, and 1 21 0, where the first-order Laue zone ring's 1 21 1 lands at 3.87 1/A)
are no reflection of the F lattice, and neither abTEM nor the kit lists them. It is sampled at 0.1 A, its pattern
reaching 3.50 1/A: inside the midpoint (3.83 1/A at 200 kV, lambda g^2 / 2 = 1 / c) past which abTEM gives an allowed
pixel to the second-order Laue zone reflection of smaller excitation error and names it so, while the kit names it by its
zero-order hkl. abTEM's
default sg_max (`estimate_necessary_excitation_error`) leaves zero-order beams beyond about sqrt(2 g_max) unindexed, so
the generator passes lambda g_max^2 / 2, the largest zero-order excitation error on the pattern.

The atoms come from the slab generator's own rules (abTEM's `orthogonalize_cell` on the turned cell, coordinates snapped,
the z shift restated): abTEM 1.0.10's infinite projection drops an atom whose coordinate is a tiny negative number, so the
fixtures are made only from snapped coordinates, as tools/pins/slab does.

The test holds every intensity at 1e-10 of the incident beam (the kit's exit-wave pin, 1e-10 of max with abTEM's float32
pi) and the hkl sets exact.

Run from the repo root:  .venv/bin/python -I -B tools/pins/beam-intensities/pins.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import platform

import numpy as np
import abtem
import ase

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "..", "Tests", "SimulatoKitParityTests", "Fixtures")

_spec = importlib.util.spec_from_file_location("slab_pins", os.path.join(HERE, "..", "slab", "pins.py"))
slab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(slab)  # sets abTEM's config: float64, numpy FFT, eager

assert abtem.__version__ == "1.0.10", abtem.__version__
assert ase.__version__ == "3.29.0", ase.__version__

ENERGY = 200e3
SLICE_THICKNESS = 1.0
SAMPLING = 0.2  # A, about: each side gets an even number of pixels
TARGET_NM = (2, 10, 25, 50, 100)
# Si [110] at 0 to 8 periods for the truth test (the forbidden 002 beam grows from zero).
TRUTH_PERIODS = (1, 2, 3, 4, 6, 8)
# (structure, zone, sampling in A); Si [001] at 0.1 A reaches as close to the Laue zone rings as the names agree.
CASES = [("Si", (1, 1, 0), SAMPLING), ("Si", (1, 1, 1), SAMPLING), ("Mg", (0, 0, 1), SAMPLING), ("Si", (0, 0, 1), 0.1)]
CENTERING = {"Si": "F", "Mg": "P"}


def unit_of(atoms, uvw):
    R, _ = slab.zone_rotation(np.array(atoms.cell), uvw)
    n, u, refused = slab.ladder(atoms, R)
    if u is None:
        raise SystemExit(f"{uvw}: refused {refused}")
    return n, u


def slab_atoms(u, periods):
    period = u["box"][2]
    k, h = slab.slice_plan(period, SLICE_THICKNESS)
    delta, margin = slab.z_shift([p[2] for p in u["positions"]], h)
    if margin < 0.05:
        raise SystemExit(f"plane margin {margin}")
    pos = np.array(u["positions"])
    pos[:, 2] = slab.to_depth((pos[:, 2] + delta) % period, period)
    numbers = u["numbers"] * periods
    positions = np.concatenate([pos + [0, 0, m * period] for m in range(periods)])
    box = [u["box"][0], u["box"][1], period * periods]
    return ase.Atoms(numbers=numbers, positions=positions, cell=box, pbc=True), k, h


class NamesDiffer(Exception):
    """abTEM names a pixel the kit selects by another (higher-order Laue zone) reflection: the pattern reaches past the
    midpoint of the excitation errors. A caller can try a coarser sampling."""


def reflects(hkl, centering):
    """The lattice's reflection condition by its centring letter (abTEM 1.0.10's get_reflection_condition for P, F and I;
    its A, B and C branches raise, so they are written out here: C forbids an odd h + k, A an odd k + l, B an odd h + l)."""
    h, k, l = (int(v) for v in hkl)
    c = centering.upper()
    if c == "P":
        return True
    if c == "F":
        return h % 2 == k % 2 == l % 2
    if c == "I":
        return (h + k + l) % 2 == 0
    if c == "C":
        return (h + k) % 2 == 0
    if c == "A":
        return (k + l) % 2 == 0
    if c == "B":
        return (h + l) % 2 == 0
    raise ValueError(centering)


def kit_beams(V, box, gpts, crop, centering):
    """The kit's rule: every hkl with (V hkl)_z = 0 that the centred lattice reflects (abTEM's `get_reflection_condition`)
    on a pixel of the cropped pattern, |g| at most the pattern's largest frequency (abTEM's default g_max, the larger of the
    two axes' (n - 1) / 2 / L)."""
    from abtem.bloch.utils import get_reflection_condition
    g_max = max((crop[0] - 1) // 2 / box[0], (crop[1] - 1) // 2 / box[1])
    Vi = np.linalg.inv(np.array(V, dtype=float))
    beams = []
    for px in range(-(crop[0] // 2), crop[0] // 2 + 1):
        for py in range(-(crop[1] // 2), crop[1] // 2 + 1):
            g = math.hypot(px / box[0], py / box[1])
            if not g <= g_max * (1 + 1e-12):
                continue
            h = Vi @ np.array([px, py, 0.0])
            r = np.round(h)
            if np.max(np.abs(h - r)) > 1e-9:
                continue
            r = r.astype(int)
            assert (np.array(V) @ r).tolist() == [px, py, 0]
            ok = reflects(r, centering)
            if centering.upper() in ("P", "F", "I"):
                assert bool(get_reflection_condition(r[None, :], centering)[0]) == ok, (r, centering)
            if not ok:
                continue
            beams.append((g, tuple(int(v) for v in r), px, py))
    beams.sort(key=lambda b: (round(b[0], 9), b[1]))
    return beams


def run_case(atoms0, uvw, periods_list, sampling, centering, energy=ENERGY):
    n, u = unit_of(atoms0, uvw)
    V = u["supercell"]
    period = u["box"][2]
    periods = max(periods_list)
    atoms, k, h = slab_atoms(u, periods)
    box = atoms.cell.lengths()
    gpts = tuple(2 * math.ceil(L / sampling / 2) for L in box[:2])
    planes = tuple(p * k - 1 for p in periods_list)
    pot = abtem.Potential(atoms, gpts=gpts, slice_thickness=[h] * (k * periods), projection="infinite",
                          exit_planes=planes)
    waves = abtem.PlaneWave(energy=energy).multislice(pot).compute()
    patterns = waves.diffraction_patterns()
    crop = patterns.array.shape[-2:]
    cell = np.linalg.inv(np.array(V, dtype=float)) @ np.diag(u["box"])  # crystal a, b, c in the box frame
    # abTEM's default sg_max (estimate_necessary_excitation_error) drops zero-order beams beyond sqrt(2 g_max) 1/A; the
    # largest zero-order excitation error on the pattern, lambda g_max^2 / 2, keeps them all.
    g_max = max(patterns.max_frequency)
    sg_max = 0.5 * abtem.core.energy.energy2wavelength(energy) * g_max ** 2 * (1 + 1e-6)
    # abTEM's own centring where its rule works (P, F, I); a C, A or B lattice is indexed as P, the kit's rule drops the rest.
    abtem_centering = centering if centering.upper() in ("P", "F", "I") else "P"
    indexed = patterns.index_diffraction_spots(cell=cell, sg_max=sg_max, centering=abtem_centering)
    abtem_hkl = np.asarray(indexed.miller_indices, dtype=int)
    abtem_values = np.asarray(indexed.array)  # (planes, spots), zero where the overlap rule gave the pixel to another
    # abTEM's own pixel of each indexed reflection (`_find_projected_pixel_index`) and its winner per pixel (the
    # smallest |excitation error|, `overlapping_spots_mask`).
    from abtem.bloch.indexing import _find_projected_pixel_index, overlapping_spots_mask
    from abtem.bloch.utils import excitation_errors, reciprocal_cell
    g_vec = abtem_hkl @ reciprocal_cell(cell)
    nm = _find_projected_pixel_index(g_vec, crop, patterns.sampling)
    winner = overlapping_spots_mask(nm, excitation_errors(g_vec, energy))
    label = {}
    for i in np.flatnonzero(winner):
        label[(int(nm[i, 0]) - crop[0] // 2, int(nm[i, 1]) - crop[1] // 2)] = i

    beams = kit_beams(V, box, gpts, crop, centering)
    intensities = np.zeros((len(planes), len(beams)))
    labels = []
    full = np.abs(np.fft.fft2(np.asarray(waves.array) / (gpts[0] * gpts[1]))) ** 2
    for j, (_, hkl, px, py) in enumerate(beams):
        if (px, py) not in label:
            raise NamesDiffer(f"{uvw}: abTEM indexes nothing on the pixel of {hkl} {px} {py}")
        i = label[(px, py)]
        intensities[:, j] = abtem_values[:, i]
        labels.append([int(v) for v in abtem_hkl[i]])
        direct = full[:, px % gpts[0], py % gpts[1]]
        # abTEM's pixel of the label is the kit's (V hkl)_xy.
        bound = 1e-15 * max(1.0, float(np.max(direct)))
        err = float(np.max(np.abs(intensities[:, j] - direct)))
        if not err <= bound:
            raise SystemExit(f"{uvw} {hkl}: abTEM's value differs from the pixel (V hkl)_xy by {err}")
    # Near the first-order Laue ring abTEM can give a pixel to a higher-order reflection of smaller excitation error; the
    # kit names it by its zero-order hkl. The pinned grids keep the ring outside the pattern, so the names agree.
    holz = [(hkl, lab) for (_, hkl, _, _), lab in zip(beams, labels) if tuple(lab) != hkl]
    if holz:
        raise NamesDiffer(f"{uvw}: abTEM names {len(holz)} pixels by another reflection, e.g. {holz[:3]}")
    thick = [p * period for p in periods_list]
    print(f"{atoms0.get_chemical_formula()} {uvw}: reps {n} V {V} box {np.round(box[:2], 4).tolist()} gpts {gpts} "
          f"crop {crop} slices {k * periods} beams {len(beams)} thick {np.round(np.array(thick) / 10, 2).tolist()} nm")
    return u, V, gpts, k, planes, thick, beams, intensities, labels


def main():
    s = slab.structures()
    pins = {"generator": "tools/pins/beam-intensities/pins.py",
            "versions": {"abtem": abtem.__version__, "ase": ase.__version__, "numpy": np.__version__,
                         "python": platform.python_version()},
            "energy": ENERGY, "slice_thickness": SLICE_THICKNESS,
            "structures": {name: {"cell": np.array(s[name].cell).tolist(),
                                  "sites": [[int(z)] + [float(v) for v in f]
                                            for z, f in zip(s[name].numbers, s[name].get_scaled_positions())]}
                           for name in ("Si", "Mg")},
            "cases": []}
    for name, uvw, sampling in CASES:
        _, u = unit_of(s[name], uvw)
        period = u["box"][2]
        periods_list = sorted({max(1, round(t * 10 / period)) for t in TARGET_NM})
        assert len(periods_list) == 5
        u, V, gpts, k, planes, thick, beams, I, labels = run_case(s[name], uvw, periods_list, sampling, CENTERING[name])
        pins["cases"].append({"structure": name, "zone": list(uvw), "supercell": V, "periods": periods_list,
                              "slices_per_period": k, "gpts": list(gpts), "exit_planes": list(planes),
                              "thicknesses": thick, "hkl": [list(b[1]) for b in beams],
                              "pixels": [[b[2], b[3]] for b in beams],
                              "intensities": I.tolist()})
        if (name, uvw) == ("Si", (1, 1, 0)):
            j = [b[1] for b in beams].index((0, 0, 2))
            print("  002:", [f"{v:.3e}" for v in I[:, j]])

    # The truth: Si [110] 002 from zero, thin.
    _, u = unit_of(s["Si"], (1, 1, 0))
    *_, beams, I, _ = run_case(s["Si"], (1, 1, 0), list(TRUTH_PERIODS), SAMPLING, CENTERING["Si"])
    j = [b[1] for b in beams].index((0, 0, 2))
    print("  thin 002:", [f"{v:.3e}" for v in I[:, j]])
    pins["truth"] = {"periods": list(TRUTH_PERIODS), "si110_002": I[:, j].tolist()}

    with open(os.path.join(OUT, "beam-intensities-pins.json"), "w") as fh:
        json.dump(pins, fh, indent=1, sort_keys=True)
        fh.write("\n")


if __name__ == "__main__":
    main()

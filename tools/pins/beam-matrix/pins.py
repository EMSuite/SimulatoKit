"""SimulatoKit beam matrix: the dynamical zone-axis beams of many crystals, zones and beam energies against abTEM 1.0.10,
so a beam the lattice forbids, a wrong hkl or a wrong intensity shows on a case no hand-picked pin holds (the reference
matrix, suite plan of 26 Nov 2026; SimulatoKit#145 and EMS#1150 were found by a walk, not a test).

Writes Tests/SimulatoKitParityTests/Fixtures/beam-matrix-pins.json. Each case is a crystal, a zone axis and a beam
energy, built and indexed exactly as tools/pins/beam-intensities/pins.py does it (the kit's slab, abTEM's
`index_diffraction_spots` with the lattice's centring, no radius, the single pixel), stored after the whole numbers of
periods nearest 5, 20 and 50 nm. The sampling is the finest of 0.1, 0.15 and 0.2 A at which abTEM names every pixel by
its zero-order hkl (a finer pattern reaches the higher-order Laue zones, where abTEM gives a pixel to the reflection of
smaller excitation error; the kit names beams of the zero-order zone only).

Crystals: fcc Al, Cu, Au; diamond Si; rock-salt MgO; bcc Fe; hcp Mg and Zr; 2H-WS2; rutile TiO2; perovskite SrTiO3.
Up to three zones each where the orthogonal unit stays small; 80, 200 and 300 kV. `fast` marks one case per crystal at 200 kV, run by every pull request; the full matrix runs in the
parity target (nightly, before a tag).

Run from the repo root:  .venv/bin/python -I -B tools/pins/beam-matrix/pins.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform

import numpy as np
from ase.build import bulk
from ase.spacegroup import crystal

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "..", "Tests", "SimulatoKitParityTests", "Fixtures")

_spec = importlib.util.spec_from_file_location("beam_pins", os.path.join(HERE, "..", "beam-intensities", "pins.py"))
beam = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(beam)
slab = beam.slab
abtem, ase = beam.abtem, beam.ase

ENERGIES = (80e3, 200e3, 300e3)
TARGET_NM = (5, 20, 50)
SAMPLINGS = (0.1, 0.15, 0.2)


def structures():
    s = {k: v for k, v in slab.structures().items() if k in ("Al", "Si", "Cu", "Fe", "Mg", "SrTiO3")}
    s["Au"] = bulk("Au", "fcc", a=4.078, cubic=True)
    s["MgO"] = bulk("MgO", "rocksalt", a=4.212, cubic=True)
    s["Zr"] = crystal("Zr", [(1 / 3, 2 / 3, 0.25)], spacegroup=194, cellpar=[3.232, 3.232, 5.147, 90, 90, 120])
    s["WS2"] = crystal(["W", "S"], [(1 / 3, 2 / 3, 0.25), (1 / 3, 2 / 3, 0.621)], spacegroup=194,
                       cellpar=[3.153, 3.153, 12.323, 90, 90, 120])
    s["TiO2"] = crystal(["Ti", "O"], [(0, 0, 0), (0.3053, 0.3053, 0)], spacegroup=136,
                        cellpar=[4.594, 4.594, 2.959, 90, 90, 90])
    return s


CENTERING = {"Al": "F", "Cu": "F", "Au": "F", "Si": "F", "MgO": "F", "Fe": "I", "Mg": "P", "Zr": "P", "WS2": "P",
             "TiO2": "P", "SrTiO3": "P"}
# (structure, zone); the first zone of each structure is its fast case (at 200 kV).
ZONES = [("Al", (0, 0, 1)), ("Al", (1, 1, 0)), ("Al", (1, 1, 2)),
         ("Cu", (1, 1, 1)),
         ("Au", (1, 1, 0)), ("Au", (0, 0, 1)),
         ("Si", (0, 0, 1)), ("Si", (1, 1, 0)), ("Si", (1, 1, 2)),
         ("MgO", (0, 0, 1)), ("MgO", (1, 1, 0)), ("MgO", (1, 1, 1)),
         ("Fe", (0, 0, 1)), ("Fe", (1, 1, 1)), ("Fe", (1, 1, 0)),
         ("Mg", (0, 0, 1)), ("Mg", (1, 0, 0)),
         ("Zr", (0, 0, 1)), ("Zr", (1, 0, 0)),
         ("WS2", (0, 0, 1)), ("WS2", (1, 0, 0)),
         ("TiO2", (0, 0, 1)), ("TiO2", (1, 1, 0)), ("TiO2", (1, 0, 0)),
         ("SrTiO3", (0, 0, 1)), ("SrTiO3", (1, 1, 0))]
# Not here: monoclinic beta'' Mg5Si6. Its [010] unit is 6.7 by 468 A (V row -32 0 -19), and abTEM 1.0.10 bounds each index
# of its hkl enumeration so that (-4 0 8), inside g_max, is not indexed; its [100] fails the slab's plane margin. A
# low-symmetry case needs a cell whose zone units stay small (to add).


def main():
    s = structures()
    names = sorted({z[0] for z in ZONES})
    pins = {"generator": "tools/pins/beam-matrix/pins.py",
            "versions": {"abtem": abtem.__version__, "ase": ase.__version__, "numpy": np.__version__,
                         "python": platform.python_version()},
            "slice_thickness": beam.SLICE_THICKNESS,
            "structures": {name: {"cell": np.array(s[name].cell).tolist(), "centering": CENTERING[name],
                                  "sites": [[int(z)] + [float(v) for v in f]
                                            for z, f in zip(s[name].numbers, s[name].get_scaled_positions())]}
                           for name in names},
            "cases": [], "skipped": []}
    seen = set()
    for name, uvw in ZONES:
        try:
            _, u = beam.unit_of(s[name], uvw)
        except SystemExit as e:
            pins["skipped"].append({"structure": name, "zone": list(uvw), "reason": str(e)})
            continue
        period = u["box"][2]
        periods_list = sorted({max(1, round(t * 10 / period)) for t in TARGET_NM})
        for energy in ENERGIES:
            result = None
            for sampling in SAMPLINGS:
                try:
                    result = beam.run_case(s[name], uvw, periods_list, sampling, CENTERING[name], energy=energy)
                    break
                except beam.NamesDiffer as e:
                    reason = str(e)
                    print(f"  {name} {uvw} {energy / 1e3:.0f} kV at {sampling} A: {e}")
                except SystemExit as e:  # the slab generator's own refusals (plane margin)
                    reason = str(e)
                    break
            if result is None:
                pins["skipped"].append({"structure": name, "zone": list(uvw), "energy": energy, "reason": reason})
                continue
            u, V, gpts, k, planes, thick, beams, I, _ = result
            fast = energy == 200e3 and name not in seen
            if fast:
                seen.add(name)
            pins["cases"].append({"structure": name, "zone": list(uvw), "energy": energy, "sampling": sampling,
                                  "fast": fast, "supercell": V, "periods": periods_list, "slices_per_period": k,
                                  "gpts": list(gpts), "thicknesses": thick, "hkl": [list(b[1]) for b in beams],
                                  "intensities": I.tolist()})
    print(f"{len(pins['cases'])} cases, {len(pins['skipped'])} skipped: {pins['skipped']}")
    with open(os.path.join(OUT, "beam-matrix-pins.json"), "w") as fh:
        json.dump(pins, fh, indent=1, sort_keys=True)
        fh.write("\n")


if __name__ == "__main__":
    main()

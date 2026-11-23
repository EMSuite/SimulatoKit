"""SimulatoKit Kirkland parameterisation: the pin generator.

abTEM 1.0.10 `KirklandParametrization` (E. J. Kirkland, Advanced Computing in Electron Microscopy, 2nd ed., Springer
2010, Appendix C), float64, infinite projection. Re-derives the projected scattering factor from the published formula
in plain NumPy (`text_*` below), asserts it equals abTEM's, then writes

  Sources/SimulatoMultislice/KirklandTable.swift            (abTEM's kirkland.json, verbatim numbers, with its sha256)
  Tests/SimulatoKitParityTests/Fixtures/kirkland-pins.json  (f(k^2) for Z = 1 ... 103 under both pi choices, the cases)
  Tests/SimulatoKitParityTests/Fixtures/kirkland-arrays.bin (float64 little-endian; complex as interleaved re, im)

The cases: the projected potential of one Al, Si, Sr, Ti and O atom, and the exit wave of the Si [110] slab of
slab-pins.json (its atoms and slices exactly, 200 kV) with `parametrization="kirkland"`. The tests hold f at 1e-14
relative, the potentials at 1e-11 of the largest value (the Lobato pin's bound) and the exit wave at the kit's multislice
pin (1e-10 with abTEM's float32 pi, 1e-6 with the exact pi).

abTEM's pi: `projected_scattering_factor` (functions/kirkland.py) takes pi as a float32 constant inside the squares,
pi**2 = 9.86960506439209 (the float32 square of the float32 pi, not the float32 pi^2 the Lobato function uses), and
the exact float64 pi everywhere else.

Run from the repo root after tools/pins/slab/pins.py:  .venv/bin/python -I -B tools/pins/kirkland/pins.py
"""

from __future__ import annotations

import hashlib
import json
import os
import platform

import numpy as np
import abtem
import ase
from ase import Atoms

assert abtem.__version__ == "1.0.10", abtem.__version__
abtem.config.set({"precision": "float64", "fft": "numpy", "dask.lazy": False, "diagnostics.progress_bar": False})

from abtem.core.constants import kappa as ABTEM_KAPPA  # noqa: E402
from abtem.parametrizations import KirklandParametrization  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
FIXTURES = os.path.join(ROOT, "Tests", "SimulatoKitParityTests", "Fixtures")
TABLE_SWIFT = os.path.join(ROOT, "Sources", "SimulatoMultislice", "KirklandTable.swift")
SYMBOLS = ase.data.chemical_symbols
KIRKLAND = KirklandParametrization()
PI32 = np.float32(np.pi)
PI2_KIRKLAND32 = float(PI32 * PI32)  # float32 arithmetic: 9.86960506439209
K2 = [0.0, 1e-4, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 9.0, 16.0, 25.0, 100.0]

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


def relmax(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = np.max(np.abs(b))
    return float(np.max(np.abs(a - b)) / (m if m > 0 else 1.0))


def check(label, value, bound):
    ok = value <= bound
    CHECKS.append(f"{'ok ' if ok else 'FAIL'} {label}: {value:.3e} <= {bound:.0e}")
    if not ok:
        print("\n".join(CHECKS))
        raise SystemExit(f"self-check failed: {label}")


# --------------------------------------------------------------------------------------------------------------------
# The formulas, written from Kirkland (2010) eq. C.17 / C.19 and abTEM's unit scaling, independently of abTEM's code.
# --------------------------------------------------------------------------------------------------------------------


def text_scaled(symbol):
    p = np.array(KIRKLAND.parameters[symbol], dtype=np.float64)  # rows a, b, c, d; three terms each
    a = np.pi * p[0] / ABTEM_KAPPA
    b = 2.0 * np.pi * np.sqrt(p[1])
    c = np.pi ** 1.5 * p[2] / p[3] ** 1.5 / ABTEM_KAPPA
    d = np.pi ** 2 / p[3]
    return a, b, c, d


def text_f(symbol, k2, pi_choice):
    """f(k^2) = sum_i 4 pi a_i / (4 pi^2 k^2 + b_i^2) + sqrt(pi / d_i) c_i pi / d_i exp(-pi^2 k^2 / d_i), term by term."""
    a, b, c, d = text_scaled(symbol)
    pi2 = PI2_KIRKLAND32 if pi_choice == "abTEMFloat32" else np.pi ** 2
    k2 = np.asarray(k2, dtype=np.float64)
    out = np.zeros_like(k2)
    for i in range(3):
        out = out + 4 * np.pi * a[i] / (4 * pi2 * k2 + b[i] ** 2)
        out = out + np.sqrt(np.pi / d[i]) * c[i] * np.pi / d[i] * np.exp(-pi2 * k2 / d[i])
    return out


def text_compensation(gpts, sampling):
    kx = np.fft.fftfreq(gpts[0], sampling[0])[:, None]
    ky = np.fft.fftfreq(gpts[1], sampling[1])[None, :]
    k = np.sqrt((kx * sampling[0]) ** 2 + (ky * sampling[1]) ** 2)
    k[0, 0] = 1.0
    s = np.sin(k) / k * sampling[0] * sampling[1]
    s[0, 0] = sampling[0] * sampling[1]
    return s


def text_single_atom(z, xy, gpts, sampling, pi_choice):
    """One atom: Re IFFT(FFT(bilinear delta) f / S)."""
    arr = np.zeros(gpts)
    u, v = xy[0] / sampling[0], xy[1] / sampling[1]
    i, j = int(np.floor(u)), int(np.floor(v))
    fx, fy = u - i, v - j
    for di, dj, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        arr[(i + di) % gpts[0], (j + dj) % gpts[1]] += w
    kx = np.fft.fftfreq(gpts[0], sampling[0])[:, None]
    ky = np.fft.fftfreq(gpts[1], sampling[1])[None, :]
    f = text_f(SYMBOLS[z], kx ** 2 + ky ** 2, pi_choice)
    return np.fft.ifft2(np.fft.fft2(arr.astype(np.complex128)) * f / text_compensation(gpts, sampling)).real


def main():
    write_table()
    pins = {"abtem": abtem.__version__, "numpy": np.__version__, "ase": ase.__version__,
            "python": platform.python_version(), "pi2_kirkland_float32": PI2_KIRKLAND32, "kappa": float(ABTEM_KAPPA),
            "f_k2": K2, "f": {}}

    # f(k^2) for every element under both pi choices.
    for z in range(1, 104):
        sym = SYMBOLS[z]
        ref = np.asarray(KIRKLAND.projected_scattering_factor(sym)(np.array(K2)), dtype=np.float64)
        mine = text_f(sym, K2, "abTEMFloat32")
        check(f"f {sym} text = abTEM", relmax(mine, ref), 1e-15)
        exact = text_f(sym, K2, "exact")
        check(f"f {sym} exact pi within 1e-7 of abTEM", relmax(exact, ref), 1e-7)
        pins["f"][sym] = {"z": z, "abtem": ref.tolist(), "exact": exact.tolist()}
        sa = np.asarray(KIRKLAND.scaled_parameters(sym, "projected_scattering_factor"))
        check(f"scaled {sym} text = abTEM", relmax(np.vstack(text_scaled(sym)), sa), 1e-15)
        pins["f"][sym]["scaled"] = sa.tolist()

    # One atom each: the projected potential.
    box, gpts, slice_thickness, xy = (6.0, 6.4, 2.0), (48, 50), 2.0, (2.13, 2.91)
    sampling = (box[0] / gpts[0], box[1] / gpts[1])
    pins["atoms"] = {"box": list(box), "gpts": list(gpts), "slice_thickness": slice_thickness, "position": [*xy, 1.0],
                     "cases": {}}
    for sym in ("O", "Al", "Si", "Ti", "Sr"):
        z = SYMBOLS.index(sym)
        atoms = Atoms(numbers=[z], positions=[[xy[0], xy[1], 1.0]], cell=box, pbc=True)
        pot = abtem.Potential(atoms, gpts=gpts, slice_thickness=slice_thickness, parametrization="kirkland",
                              projection="infinite")
        v = np.asarray(pot.build().compute().array, dtype=np.float64)
        check(f"{sym} one slice", float(abs(v.shape[0] - 1)), 0)
        check(f"{sym} V text = abTEM", relmax(text_single_atom(z, xy, gpts, sampling, "abTEMFloat32"), v[0]), 1e-12)
        put(f"atom.{sym}.V", v[0])
        pins["atoms"]["cases"][sym] = {"z": z, "V_max_abs": float(np.max(np.abs(v[0]))), "V_sum": float(v[0].sum())}

    # The Si [110] slab of slab-pins.json with Kirkland's parameters.
    slab = json.load(open(os.path.join(FIXTURES, "slab-pins.json")))["slab"]
    atoms = Atoms(numbers=slab["numbers"], positions=slab["positions"], cell=slab["box"], pbc=True)
    thicknesses = [slab["slice_height"]] * slab["slices"]
    pot = abtem.Potential(atoms, gpts=tuple(slab["gpts"]), slice_thickness=thicknesses, parametrization="kirkland",
                          projection="infinite")
    exit_wave = np.asarray(abtem.PlaneWave(energy=slab["energy"]).multislice(pot).compute().array)
    lobato = np.asarray(abtem.PlaneWave(energy=slab["energy"]).multislice(
        abtem.Potential(atoms, gpts=tuple(slab["gpts"]), slice_thickness=thicknesses, projection="infinite")
    ).compute().array)
    d = relmax(exit_wave, lobato)
    check("Kirkland and Lobato exit waves differ (parameterisation reached the run)", -d, -1e-4)
    put("si110.exit", exit_wave)
    pins["slab"] = {"source": "slab-pins.json", "slices": slab["slices"], "slice_height": slab["slice_height"],
                    "gpts": slab["gpts"], "energy": slab["energy"], "exit_max_abs": float(np.max(np.abs(exit_wave))),
                    "kirkland_vs_lobato_relmax": d}

    blob = bytes(BLOB)
    pins["arrays"] = {"file": "kirkland-arrays.bin", "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob),
                      "index": INDEX}
    pins["checks"] = CHECKS
    with open(os.path.join(FIXTURES, "kirkland-arrays.bin"), "wb") as fh:
        fh.write(blob)
    with open(os.path.join(FIXTURES, "kirkland-pins.json"), "w") as fh:
        json.dump(pins, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"wrote {len(INDEX)} arrays, {len(blob) / 1e3:.1f} KB; {len(CHECKS)} self-checks passed; "
          f"Kirkland vs Lobato exit wave {d:.3e}")


def write_table():
    path = os.path.join(os.path.dirname(abtem.parametrizations.__file__), "data", "kirkland.json")
    raw = open(path, "rb").read()
    data = json.loads(raw)
    lines = [
        "//",
        "//  KirklandTable.swift",
        "//  Role: Kirkland's (2010) parameters (a_i, b_i, c_i, d_i), i = 1 ... 3, per element Z = 1 ... 103, as abTEM",
        "//        1.0.10 ships them (abtem/parametrizations/data/kirkland.json, sha256 " + hashlib.sha256(raw).hexdigest()[:16]
        + "...).",
        "//        Generated by tools/pins/kirkland/pins.py; do not edit. Numbers verbatim (Python repr of each double).",
        "//        E. J. Kirkland, Advanced Computing in Electron Microscopy, 2nd ed., Springer (2010), Appendix C.",
        "//        abTEM is GPL-3.0-or-later (NOTICE).",
        "//",
        "",
        "/// `kirklandTable[Z - 1]` = (a, b, c, d), three terms each; `nil` where abTEM's table has no entry.",
        "let kirklandTable: [(a: [Double], b: [Double], c: [Double], d: [Double])?] = [",
    ]
    for z in range(1, 104):
        sym = SYMBOLS[z]
        if sym in data:
            rows = [", ".join(repr(float(x)) for x in r) for r in data[sym]]
            assert len(rows) == 4 and all(len(r) == 3 for r in data[sym]), sym
            lines.append(f"    (a: [{rows[0]}], b: [{rows[1]}], c: [{rows[2]}], d: [{rows[3]}]), // {z} {sym}")
        else:
            lines.append(f"    nil, // {z} {sym}")
    lines += ["]", "", "/// sha256 of abTEM's kirkland.json the table was generated from.",
              f"let kirklandTableSourceSHA256 = \"{hashlib.sha256(raw).hexdigest()}\"", ""]
    with open(TABLE_SWIFT, "w") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()

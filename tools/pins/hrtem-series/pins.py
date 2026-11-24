"""SimulatoKit HRTEM image series: the pin generator.

abTEM 1.0.10 (float64, numpy FFT, eager): a SrTiO3 [001] slab with exit planes every four slices, then for each exit
plane the images over a list of defocus values, `waves.apply_ctf(CTF(defocus=..., Cs=..., focal_spread=...)).intensity()`,
and one series tiled with `Images.tile`. Each image is re-derived in NumPy from the CTF kernel and asserted equal to
abTEM's. Writes Tests/SimulatoKitParityTests/Fixtures/hrtem-series-pins.json and hrtem-series-arrays.bin.

Run from the repo root:  .venv/bin/python -I -B tools/pins/hrtem-series/pins.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIXTURES = os.path.join(ROOT, "Tests", "SimulatoKitParityTests", "Fixtures")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "tools", "pins", path, "pins.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s1 = load("s1", "potential-multislice")
s3 = load("s3", "ctf")

import abtem  # noqa: E402
from ase import Atoms  # noqa: E402

BLOB = bytearray()
INDEX: dict = {}
CHECKS: list[str] = []


def put(name, a):
    a = np.ascontiguousarray(a)
    INDEX[name] = {"offset": len(BLOB), "count": int(a.size), "shape": list(a.shape), "kind": "f8"}
    BLOB.extend(a.astype("<f8").tobytes())


def check(label, value, bound):
    ok = value <= bound
    CHECKS.append(f"{'ok ' if ok else 'FAIL'} {label}: {value:.3e} <= {bound:.0e}")
    if not ok:
        print("\n".join(CHECKS))
        raise SystemExit(f"self-check failed: {label}")


def main():
    a0 = 3.905
    sto = Atoms("SrTiO3", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
                cell=[a0, a0, a0], pbc=True) * (2, 2, 5)
    gpts, energy, slice_thickness, every = (64, 64), 300e3, 1.9525, 4
    extent = (2 * a0, 2 * a0)
    cs, focal_spread, cutoff = 1.2e7, 30.0, 20.0
    defocus = [-200.0, 0.0, 150.0, 400.0]
    tile = (2, 3)

    pot = abtem.Potential(sto, gpts=gpts, slice_thickness=slice_thickness, exit_planes=every)
    exits = abtem.PlaneWave(energy=energy).multislice(pot).compute()
    psi = np.asarray(exits.array)  # (planes, nx, ny), the incident wave first
    planes = psi.shape[0]
    check("exit planes: incident, after 4, 8, and the last of 10", abs(planes - 4), 0)

    for p in range(planes):
        wave = exits[p]
        for d, df in enumerate(defocus):
            ctf = abtem.CTF(defocus=df, Cs=cs, focal_spread=focal_spread, semiangle_cutoff=cutoff)
            img = np.asarray(wave.apply_ctf(ctf).intensity().compute().array)
            kt = s3.text_kernel(gpts, extent, energy, {"C10": -df, "C30": cs}, cutoff, True, focal_spread, 0.0)
            timg = np.abs(np.fft.ifft2(np.fft.fft2(psi[p]) * kt)) ** 2
            check(f"plane {p} defocus {df} image text vs abTEM", s1.relmax(timg, img), 1e-12)
            put(f"plane{p}.defocus{d}", img)
    # One tiled series: the last plane at every defocus, tiled (2, 3).
    for d, df in enumerate(defocus):
        ctf = abtem.CTF(defocus=df, Cs=cs, focal_spread=focal_spread, semiangle_cutoff=cutoff)
        tiled = np.asarray(exits[planes - 1].apply_ctf(ctf).intensity().tile(tile).compute().array)
        check(f"tiled defocus {df} is the image repeated", s1.relmax(np.tile(tiled[:gpts[0], :gpts[1]], tile),
                                                                   tiled), 0)
        put(f"tiled.defocus{d}", tiled)

    pins = {"versions": {"abtem": abtem.__version__, "numpy": np.__version__},
            "structure": "SrTiO3 [001], 2 x 2 x 5 cells, a = 3.905", "gpts": list(gpts), "extent": list(extent),
            "energy": energy, "slice_thickness": slice_thickness, "exit_planes_every": every, "planes": planes,
            "Cs": cs, "focal_spread": focal_spread, "semiangle_cutoff": cutoff, "defocus": defocus, "tile": list(tile)}
    os.makedirs(FIXTURES, exist_ok=True)
    blob = bytes(BLOB)
    open(os.path.join(FIXTURES, "hrtem-series-arrays.bin"), "wb").write(blob)
    pins["arrays"] = {"file": "hrtem-series-arrays.bin", "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob),
                      "index": INDEX}
    pins["self_checks"] = CHECKS
    json.dump(pins, open(os.path.join(FIXTURES, "hrtem-series-pins.json"), "w"), indent=1, sort_keys=True)
    print("\n".join(CHECKS))
    print(f"wrote {len(INDEX)} arrays, {len(blob) / 1e6:.2f} MB; {len(CHECKS)} self-checks passed")


if __name__ == "__main__":
    main()

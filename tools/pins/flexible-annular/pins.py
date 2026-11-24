"""SimulatoKit flexible annular detector: the pin generator (MK10).

Runs abTEM 1.0.10 in float64 (numpy FFT, eager): a probe scanned over a 4 x 4 grid of Si [110] (the kit's slab) and of
SrTiO3 [001], recorded by FlexibleAnnularDetector (1 mrad bins), and the AnnularDetector of the same limits. Each bin is
re-derived from the exit wave in plain NumPy (text_bins: the mask of text_annular in tools/pins/stem-detectors/pins.py,
one per bin) and asserted equal to abTEM's; the sum of the bins from `inner` to `outer` is asserted equal to
AnnularDetector(inner, outer). A failure stops the generator. Writes

  Tests/SimulatoKitParityTests/Fixtures/flexible-annular-pins.json
  Tests/SimulatoKitParityTests/Fixtures/flexible-annular-arrays.bin   (float64 little-endian)

abTEM returns the bins as float32, so the arrays are held at float32 bounds (tools/pins/float32-arrays.txt) and a kit
test uses the float32 tolerance of the STEM detector pin (tools/pins/stem-detectors) for the bins. Atoms come from snapped coordinates (abTEM 1.0.10's infinite-projection bug, as tools/pins/slab).

Run from the repo root:  .venv/bin/python -I -B tools/pins/flexible-annular/pins.py
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
from ase import Atoms  # noqa: E402

ENERGY = 200e3
CUTOFF = 20.0  # mrad, the probe's semiangle
STEP = 1.0  # mrad, the bins
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


def text_bins(psi, gpts, sampling, lam, inner, nbins, step):
    """The intensity of the far field in each radial bin [inner + i step, inner + (i + 1) step) mrad."""
    p = np.abs(np.fft.fft2(psi)) ** 2
    da = s2.angular_sampling(gpts, sampling, lam)
    ax = np.fft.fftfreq(gpts[0], 1 / da[0] / gpts[0])[:, None]
    ay = np.fft.fftfreq(gpts[1], 1 / da[1] / gpts[1])[None, :]
    a2 = ax ** 2 + ay ** 2
    return np.array([float((p * ((a2 >= (inner + i * step) ** 2) & (a2 < (inner + (i + 1) * step) ** 2))).sum())
                     for i in range(nbins)])


def case(name, atoms, sampling_target, slice_thickness, inner, outer, scan_gpts=(4, 4)):
    box = tuple(float(x) for x in np.diag(atoms.cell))
    gpts = tuple(2 * int(np.ceil(L / sampling_target / 2)) for L in box[:2])
    sampling = (box[0] / gpts[0], box[1] / gpts[1])
    lam = s1.energy2wavelength(ENERGY)
    pot = abtem.Potential(atoms, gpts=gpts, slice_thickness=slice_thickness, projection="infinite")
    gs = abtem.GridScan(start=[0.0, 0.0], end=[box[0], box[1]], gpts=list(scan_gpts), endpoint=False)
    probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=CUTOFF)
    flex = abtem.FlexibleAnnularDetector(step_size=STEP, inner=inner, outer=outer)
    ann = abtem.AnnularDetector(inner=inner, outer=outer)
    exits = probe.multislice(pot, scan=gs).compute()
    waves = np.asarray(exits.array).reshape(-1, *gpts)
    bins = np.asarray(flex.detect(exits).compute().array)[..., 0]  # the azimuthal axis has one bin
    annular = np.asarray(ann.detect(exits).compute().array)
    nbins = int(np.floor(outer - inner) / STEP)
    assert bins.shape[-1] == nbins, (bins.shape, nbins)
    flat = bins.reshape(-1, nbins)
    scale = float(np.abs(flat).max())
    text = np.stack([text_bins(w, gpts, sampling, lam, inner, nbins, STEP) for w in waves])
    # abTEM 1.0.10 returns the flexible detector's bins as float32 even in float64 mode (the exit wave and its diffraction
    # patterns are float64: the sum of the text bins equals the patterns' sum exactly), so the bound is float32 rounding.
    check(f"{name} bins text vs abTEM (float32 output)", float(np.abs(text - flat).max() / scale), 3e-7)
    check(f"{name} sum of bins vs AnnularDetector({inner}, {outer})",
          float(np.abs(flat.sum(-1) - annular.reshape(-1)).max() / np.abs(annular).max()), 1e-6)
    put(f"{name}.bins", flat)
    put(f"{name}.annular", annular.reshape(-1))
    put(f"{name}.scan_positions", np.asarray(gs.get_positions()))
    return {"box": list(box), "gpts": list(gpts), "energy": ENERGY, "slice_thickness": slice_thickness,
            "semiangle_cutoff_mrad": CUTOFF, "step_mrad": STEP, "inner_mrad": inner, "outer_mrad": outer,
            "nbins": nbins, "scan_gpts": list(scan_gpts), "numbers": [int(z) for z in atoms.numbers],
            "positions": atoms.positions.tolist()}


def main():
    s = bi.slab.structures()
    _, u = bi.unit_of(s["Si"], (1, 1, 0))
    si110, _, h = bi.slab_atoms(u, 2)
    a0 = 3.905
    sto = Atoms("SrTiO3", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
                cell=[a0, a0, a0], pbc=True) * (2, 2, 4)
    pins = {"generator": "tools/pins/flexible-annular/pins.py",
            "versions": {"abtem": abtem.__version__, "ase": ase.__version__, "numpy": np.__version__,
                         "python": platform.python_version()},
            "cases": {}}
    # 0.1 A sampling: the grid's largest angle is about 125 mrad, so the bins stop at 120.
    pins["cases"]["M1"] = case("M1", si110, 0.1, h, 0.0, 120.0)
    pins["cases"]["M2"] = case("M2", sto, 0.1, a0 / 2, 0.0, 120.0)
    pins["cases"]["M2_offset"] = case("M2_offset", sto, 0.1, a0 / 2, 30.0, 90.0)

    os.makedirs(FIXTURES, exist_ok=True)
    blob = bytes(BLOB)
    with open(os.path.join(FIXTURES, "flexible-annular-arrays.bin"), "wb") as fh:
        fh.write(blob)
    pins["arrays"] = {"file": "flexible-annular-arrays.bin", "sha256": hashlib.sha256(blob).hexdigest(),
                      "bytes": len(blob), "index": INDEX}
    pins["self_checks"] = CHECKS
    with open(os.path.join(FIXTURES, "flexible-annular-pins.json"), "w") as fh:
        json.dump(pins, fh, indent=1, sort_keys=True)
    print("\n".join(CHECKS))
    print(f"wrote {len(INDEX)} arrays, {len(blob) / 1e6:.2f} MB; {len(CHECKS)} self-checks passed")


if __name__ == "__main__":
    main()

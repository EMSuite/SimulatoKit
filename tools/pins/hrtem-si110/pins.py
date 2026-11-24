"""SimulatoKit Si [110] HRTEM image chain: the pin generator (MT1, MA2).

abTEM 1.0.10 (float64, numpy FFT, eager): the kit's Si [110] slab (tools/pins/slab, snapped coordinates), a plane wave
at 200 kV, exit planes at about 2, 5 and 10 nm, and for each the image at Scherzer defocus with partial coherence:
`waves.apply_ctf(CTF(defocus="scherzer", Cs=..., focal_spread=..., angular_spread=..., semiangle_cutoff=...)).intensity()`.
The whole chain is re-derived in NumPy (the potential of one period tiled, text_multislice, the CTF kernel with its
temporal and spatial envelopes) and asserted equal to abTEM's at 1e-10 of the maximum; abTEM's Scherzer defocus is asserted
equal to sqrt(3/2 Cs lambda). Writes

  Tests/SimulatoKitParityTests/Fixtures/hrtem-si110-pins.json
  Tests/SimulatoKitParityTests/Fixtures/hrtem-si110-arrays.bin   (float64 little-endian)

Run from the repo root:  .venv/bin/python -I -B tools/pins/hrtem-si110/pins.py
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
s3 = load("ctf")
bi = load("beam-intensities")

import abtem  # noqa: E402
import ase  # noqa: E402

ENERGY = 200e3
CS = 5e6  # A (0.5 mm)
FOCAL_SPREAD = 40.0  # A
ANGULAR_SPREAD = 0.2  # mrad
CUTOFF = 40.0  # mrad, the objective aperture
SAMPLING = 0.1
TARGET_NM = (2, 5, 10)
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


def main():
    s = bi.slab.structures()
    _, u = bi.unit_of(s["Si"], (1, 1, 0))
    period_atoms, k, h = bi.slab_atoms(u, 1)
    box = tuple(float(x) for x in np.diag(period_atoms.cell))
    period = box[2]
    gpts = tuple(2 * int(np.ceil(L / SAMPLING / 2)) for L in box[:2])
    sampling = (box[0] / gpts[0], box[1] / gpts[1])
    extent = (box[0], box[1])
    periods = sorted({max(1, round(t * 10 / period)) for t in TARGET_NM})
    assert len(periods) == 3, periods
    n = max(periods)
    atoms = ase.Atoms(numbers=list(period_atoms.numbers) * n,
                      positions=np.concatenate([period_atoms.positions + [0, 0, m * period] for m in range(n)]),
                      cell=[box[0], box[1], period * n], pbc=True)
    planes = tuple(p * k - 1 for p in periods)
    pot = abtem.Potential(atoms, gpts=gpts, slice_thickness=[h] * (k * n), projection="infinite", exit_planes=planes)
    exits = abtem.PlaneWave(energy=ENERGY).multislice(pot).compute()
    psi_abtem = np.asarray(exits.array)
    assert psi_abtem.shape[0] == len(planes)

    lam = s1.energy2wavelength(ENERGY)
    scherzer = float(np.sqrt(1.5 * CS * lam))
    ctf = abtem.CTF(energy=ENERGY, Cs=CS, defocus="scherzer", focal_spread=FOCAL_SPREAD, angular_spread=ANGULAR_SPREAD,
                    semiangle_cutoff=CUTOFF)
    check("Scherzer defocus vs sqrt(3/2 Cs lambda) (relative)", abs(float(ctf.defocus) - scherzer) / scherzer, 1e-12)

    positions, thick, labels = s1.text_wrap_and_slice(period_atoms.numbers, period_atoms.positions, box, h)
    slices = []
    for j in range(len(thick)):
        nums = [int(z) for z, l in zip(period_atoms.numbers, labels) if l == j]
        pj = [p for p, l in zip(positions, labels) if l == j]
        slices.append(s1.text_potential_slice(nums, pj, gpts, sampling, "abTEMFloat32"))
    assert len(slices) == k
    text = s1.text_multislice(np.ones(gpts, complex), slices * n, list(thick) * n, ENERGY, gpts, sampling, set(planes))
    kernel = s3.text_kernel(gpts, extent, ENERGY, {"C10": -scherzer, "C30": CS}, CUTOFF, True, FOCAL_SPREAD, ANGULAR_SPREAD)
    thicknesses = [p * period / 10 for p in periods]  # nm
    images = []
    for i, (t, w) in enumerate(zip(text, psi_abtem)):
        check(f"{thicknesses[i]:.1f} nm exit wave text vs abTEM", s1.relmax(t, w), 1e-10)
        img = np.asarray(exits[i].apply_ctf(ctf).intensity().compute().array)
        timg = np.abs(np.fft.ifft2(np.fft.fft2(t) * kernel)) ** 2
        check(f"{thicknesses[i]:.1f} nm image text vs abTEM", s1.relmax(timg, img), 1e-10)
        images.append(img)
    put("exit_waves", psi_abtem)
    put("images", np.stack(images))
    pins = {"generator": "tools/pins/hrtem-si110/pins.py",
            "versions": {"abtem": abtem.__version__, "ase": ase.__version__, "numpy": np.__version__,
                         "python": platform.python_version()},
            "structure": "Si [110], the kit's slab unit", "period_box": list(box), "gpts": list(gpts),
            "slice_thickness": h, "slices_per_period": k, "periods": periods, "thickness_nm": thicknesses,
            "exit_planes": list(planes), "energy": ENERGY, "Cs": CS, "defocus": scherzer, "focal_spread": FOCAL_SPREAD,
            "angular_spread": ANGULAR_SPREAD, "semiangle_cutoff": CUTOFF,
            "numbers": [int(z) for z in period_atoms.numbers], "positions": period_atoms.positions.tolist()}
    os.makedirs(FIXTURES, exist_ok=True)
    blob = bytes(BLOB)
    with open(os.path.join(FIXTURES, "hrtem-si110-arrays.bin"), "wb") as fh:
        fh.write(blob)
    pins["arrays"] = {"file": "hrtem-si110-arrays.bin", "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob),
                      "index": INDEX}
    pins["self_checks"] = CHECKS
    with open(os.path.join(FIXTURES, "hrtem-si110-pins.json"), "w") as fh:
        json.dump(pins, fh, indent=1, sort_keys=True)
    print("\n".join(CHECKS))
    print(f"wrote {len(INDEX)} arrays, {len(blob) / 1e6:.2f} MB; {len(CHECKS)} self-checks passed")


if __name__ == "__main__":
    main()

"""SimulatoKit CTF profiles and the CTF's own closed forms: the pin generator.

abTEM 1.0.10 (float64): `CTF.profiles(gpts, max_angle, phi)` for CTFs with and without an aperture and envelopes, the
phase term alone (`Aberrations` at the same angles, imaginary part of exp(-i chi)), and the instance forms
`CTF.scherzer_defocus`, `.point_resolution` and `.crossover_angle`. Every profile is re-derived from the published
formulas in NumPy and asserted equal to abTEM's. Writes Tests/SimulatoKitParityTests/Fixtures/ctf-profiles-pins.json.

Run from the repo root:  .venv/bin/python -I -B tools/pins/ctf-profiles/pins.py
"""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIXTURES = os.path.join(ROOT, "Tests", "SimulatoKitParityTests", "Fixtures")
spec = importlib.util.spec_from_file_location("s3", os.path.join(ROOT, "tools", "pins", "ctf", "pins.py"))
s3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s3)

import abtem  # noqa: E402
from abtem.core.energy import energy2wavelength  # noqa: E402

abtem.config.set({"precision": "float64"})

CHECKS: list[str] = []


def check(label, value, bound):
    ok = value <= bound
    CHECKS.append(f"{'ok ' if ok else 'FAIL'} {label}: {value:.3e} <= {bound:.0e}")
    if not ok:
        print("\n".join(CHECKS))
        raise SystemExit(f"self-check failed: {label}")


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.max(np.abs(b))
    return float(np.max(np.abs(a - b)) / (m if m > 0 else 1))


# id, energy [eV], aberrations [Å, rad], cutoff [mrad] or None, focal spread [Å], angular spread [mrad], gpts,
# max_angle [mrad] or None, phi [rad]
CASES = [
    ("P1", 200e3, {"C30": 1e7}, None, 0.0, 0.0, 1000, None, 0.0),  # Cs only, abTEM's defaults
    ("P2", 200e3, {"C30": 1e7}, 20.0, 50.0, 0.0, 400, None, 0.0),
    ("P3", 300e3, {"C30": -1e5, "C10": 40.0, "C50": 5e7}, 30.0, 30.0, 0.5, 257, None, 0.0),
    ("P4", 80e3, {"C30": 1.2e7, "C10": -400.0, "C12": 50.0, "phi12": 0.3, "C23": 2e4, "phi23": 1.1}, None, 40.0,
     0.3, 300, 35.0, 0.7),
]


def main():
    pins = {"generator": "tools/pins/ctf-profiles/pins.py", "abtem": abtem.__version__, "cases": {}, "closed_forms": []}
    for cid, energy, ab, cutoff, delta, beta, gpts, max_angle, phi in CASES:
        ab = dict(ab)
        if cid == "P2":  # defocus at Scherzer, set from abTEM's instance form
            ab["C10"] = -abtem.CTF(Cs=ab["C30"], energy=energy).scherzer_defocus
        kwargs = dict(aberration_coefficients=ab, energy=energy, focal_spread=delta, angular_spread=beta)
        if cutoff is not None:
            kwargs["semiangle_cutoff"] = cutoff
        ctf = abtem.CTF(**kwargs)
        prof = ctf.profiles(gpts=gpts, max_angle=max_angle, phi=phi)
        arr = np.asarray(prof.array, float)
        names = list(prof.ensemble_axes_metadata[-1].values) if arr.ndim == 2 else ["ctf"]
        if arr.ndim == 1:
            arr = arr[None]
        rows = dict(zip(names, arr))
        lam = energy2wavelength(energy)
        ma = max_angle if max_angle is not None else (50.0 if cutoff is None else 1.6 * cutoff)
        alpha = np.linspace(0, ma * 1e-3, gpts)
        phase = ctf._aberrations._evaluate_from_angular_grid(alpha, np.array(phi)).imag
        # Re-derive from the published formulas.
        text_phase = -np.sin(s3.text_chi(alpha, phi, ab, lam))
        check(f"{cid} phase vs text", rel(phase, text_phase), 1e-12)
        product = text_phase.copy()
        if beta != 0:
            se = s3.text_spatial(alpha, phi, ab, lam, beta)
            check(f"{cid} spatial vs text", rel(rows["spatial envelope"], se), 1e-12)
            product = product * se
        if delta != 0:
            te = s3.text_temporal(alpha, lam, delta)
            check(f"{cid} temporal vs text", rel(rows["temporal envelope"], te), 1e-12)
            product = product * te
        if cutoff is not None:
            ap = (alpha <= cutoff * 1e-3).astype(float)
            check(f"{cid} aperture vs text", rel(rows["aperture"], ap), 0)
            product = product * ap
        check(f"{cid} ctf vs text", rel(rows["ctf"], product), 1e-12)
        check(f"{cid} sampling", abs(float(np.atleast_1d(prof.sampling)[0]) - ma / (gpts - 1) / (lam * 1e3)) / float(np.atleast_1d(prof.sampling)[0]), 1e-15)
        pins["cases"][cid] = {
            "energy": energy, "aberrations": ab, "cutoff_mrad": cutoff, "focal_spread": delta, "angular_spread": beta,
            "points": gpts, "max_angle": max_angle, "phi": phi, "sampling": float(float(np.atleast_1d(prof.sampling)[0])),
            "components": names, "phase": phase.tolist(),
            **{k.replace(" ", "_"): v.tolist() for k, v in rows.items()},
        }
    for cs in (1e7, 1.2e7, -1e5, 5e4):
        for e in (80e3, 200e3, 300e3):
            ctf = abtem.CTF(Cs=cs, energy=e)
            row = {"Cs": cs, "energy": e, "scherzer": float(ctf.scherzer_defocus),
                   "point_resolution": float(ctf.point_resolution), "crossover_angle": float(ctf.crossover_angle)}
            lam = energy2wavelength(e)
            check(f"crossover {cs} {e}", abs(row["crossover_angle"] - 1e3 * lam / row["point_resolution"])
                  / row["crossover_angle"], 1e-15)
            pins["closed_forms"].append(row)
    try:
        abtem.CTF(Cs=0.0, energy=200e3).scherzer_defocus
        raise SystemExit("abTEM no longer refuses Scherzer defocus at Cs = 0")
    except ValueError:
        CHECKS.append("ok  Cs = 0 refused by scherzer_defocus")
    pins["checks"] = CHECKS
    os.makedirs(FIXTURES, exist_ok=True)
    with open(os.path.join(FIXTURES, "ctf-profiles-pins.json"), "w") as f:
        json.dump(pins, f, indent=1, sort_keys=True)
        f.write("\n")
    print("\n".join(CHECKS))


if __name__ == "__main__":
    main()

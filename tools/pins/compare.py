"""Compare the references a regeneration wrote with the committed ones (the Linux reference job, .github/workflows/references.yml).

Run from the repo root after every generator has written in place:

    python3 -I -B tools/pins/compare.py [--rtol R] [--atol A]

It reads `git status` for the changed, new and deleted files under Tests/ and Sources/ and judges each one:

  same            the bytes are equal (git lists no change)
  float noise     the file parses the same and every number agrees within the bounds below; strings that hold numbers
                  are split into text and numbers and compared the same way; the keys in VOLATILE (a run's date,
                  interpreter build, an array file's sha256) are not compared
  DIFFERS         anything else: a key, a length, a word, an integer or a number outside the bounds; a new file a generator
                  wrote that is not committed; a committed file a generator removed

The job fails on any DIFFERS. A path listed in tools/pins/known-differences.txt (each line with its reason and issue) is
reported as known and does not fail the job; the list empties as those issues close. Float noise is expected: the committed references were written on a Mac (Accelerate's libm and
FFT) and the job runs on Linux (glibc, pocketfft), which differ in the last bits of a float64. The bounds are far below any
tolerance a test holds (the tightest is assert_allclose's rtol 1e-7) and far above that noise (about 1e-15 relative):

    |a - b| <= atol + rtol * max(|a|, |b|),   rtol 1e-9, atol 1e-12

For a binary array file (`*.bin`, SimulatoKit's fixtures) the index in the JSON beside it gives each array's offset, count
and kind; atol is then taken relative to the array's largest magnitude, as a small value inside a large field carries the
field's rounding (atol also holds on its own, so an array of pure rounding residue, 1e-32 where the physics says zero, is
noise). An array that tools/pins/float32-arrays.txt names (a float32 computation stored as float64, each line with its
reason) is held at float32 bounds, rtol 1e-6 and atol 1e-7 of its largest magnitude (about ten float32
roundings).

A generator's passed self-check line ("ok  <check>: <measured> <= <bound>") compares by its words only: the measured residual
moves with the platform, the verdict and the bound do not.
"""
import argparse
import fnmatch
import json
import math
import os
import re
import subprocess
import sys

VOLATILE = {"generated_at", "generatedAt", "python", "platform", "machine", "sha256", "date"}
NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|[-+]?(?:inf|nan)\b")
RTOL32, ATOL32 = 1e-6, 1e-7
PASSED_CHECK = re.compile(r"^ok\s")
KINDS = {"f8": "<f8", "f4": "<f4", "c16": "<c16", "c8": "<c8", "i8": "<i8", "i4": "<i4", "i2": "<i2", "u1": "u1", "u2": "<u2",
         "u4": "<u4", "b1": "?"}


def close(a, b, rtol, atol):
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def compare_text(a, b, rtol, atol, where, out):
    """Text with numbers in it: the words must be equal, the numbers within the bounds."""
    if a == b:
        return
    if PASSED_CHECK.match(a) and PASSED_CHECK.match(b) and NUMBER.split(a) == NUMBER.split(b):
        return  # a generator's passed self-check: the measured residual moves with the platform, the verdict does not
    sa, sb = NUMBER.split(a), NUMBER.split(b)
    na, nb = NUMBER.findall(a), NUMBER.findall(b)
    if sa != sb or len(na) != len(nb):
        out.append(f"{where}: text differs: {a[:120]!r} vs {b[:120]!r}")
        return
    for x, y in zip(na, nb):
        if x == y:
            continue
        fx, fy = float(x), float(y)
        integral = not any(c in x + y for c in ".eEn")
        if integral or not close(fx, fy, rtol, atol):
            out.append(f"{where}: {x} vs {y}")
            return


def compare_json(a, b, rtol, atol, where, out):
    if len(out) > 20:
        return
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        if a != b:
            out.append(f"{where}: {a!r} vs {b!r}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            if a != b:
                out.append(f"{where}: {a} vs {b}")
        elif not close(float(a), float(b), rtol, atol):
            out.append(f"{where}: {a!r} vs {b!r}")
    elif isinstance(a, str) and isinstance(b, str):
        compare_text(a, b, rtol, atol, where, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{where}: length {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            compare_json(x, y, rtol, atol, f"{where}[{i}]", out)
    elif isinstance(a, dict) and isinstance(b, dict):
        ka, kb = set(a) - VOLATILE, set(b) - VOLATILE
        if ka != kb:
            out.append(f"{where}: keys {sorted(ka ^ kb)[:6]} on one side only")
            return
        for k in sorted(ka):
            compare_json(a[k], b[k], rtol, atol, f"{where}.{k}", out)
    else:
        out.append(f"{where}: {type(a).__name__} vs {type(b).__name__}")


def committed(path):
    return subprocess.run(["git", "show", f"HEAD:{path}"], capture_output=True, check=True).stdout


def array_index(path):
    """The index of a binary array file: the JSON in the same folder whose "arrays" block names it."""
    folder, name = os.path.dirname(path), os.path.basename(path)
    for other in sorted(os.listdir(folder or ".")):
        if not other.endswith(".json"):
            continue
        try:
            block = json.loads(committed(os.path.join(folder, other))).get("arrays")
        except (subprocess.CalledProcessError, ValueError, AttributeError):
            continue
        if isinstance(block, dict) and block.get("file") == name:
            return block["index"]
    return None


def listed(name):
    """The paths of a list file next to this script: one per line, each with its reason after a #."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [line.split("#", 1)[0].strip() for line in f if line.split("#", 1)[0].strip()]


def float32_arrays():
    """tools/pins/float32-arrays.txt: `<file>:<array name glob>` per line, each with its reason after a #."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "float32-arrays.txt")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        lines = [line.split("#", 1)[0].strip() for line in f]
    return [tuple(line.split(":", 1)) for line in lines if line]


def compare_arrays(path, new, old, rtol, atol, out):
    import numpy as np
    index = array_index(path)
    if index is None:
        out.append(f"{path}: binary file with no array index beside it")
        return
    if len(new) != len(old):
        out.append(f"{path}: {len(old)} bytes committed, {len(new)} written")
        return
    for name, entry in index.items():
        dtype = np.dtype(KINDS.get(entry["kind"], entry["kind"]))
        a = np.frombuffer(new, dtype=dtype, count=entry["count"], offset=entry["offset"])
        b = np.frombuffer(old, dtype=dtype, count=entry["count"], offset=entry["offset"])
        if dtype.kind in "fc":
            scale = float(np.max(np.abs(b))) if b.size else 0.0
            r, t = rtol, atol
            if any(os.path.basename(path) == f and fnmatch.fnmatchcase(name, g) for f, g in float32_arrays()):
                r, t = RTOL32, ATOL32  # a float32 result stored as float64 carries float32 rounding
            same_nan = np.isnan(a) == np.isnan(b)
            ok = same_nan & ((a == b) | (np.abs(a - b) <= atol + t * scale + r * np.maximum(np.abs(a), np.abs(b))))
            if not np.all(ok):
                i = int(np.argmin(ok))
                out.append(f"{path}:{name}[{i}]: {b[i]!r} committed, {a[i]!r} written")
        elif not np.array_equal(a, b):
            out.append(f"{path}:{name}: integer values differ")


def judge(path, rtol, atol):
    out = []
    old = committed(path)
    with open(path, "rb") as f:
        new = f.read()
    if path.endswith(".json"):
        try:
            compare_json(json.loads(new), json.loads(old), rtol, atol, path, out)
        except ValueError as error:
            out.append(f"{path}: does not parse ({error})")
    elif path.endswith(".bin"):
        compare_arrays(path, new, old, rtol, atol, out)
    else:
        try:
            compare_text(new.decode(), old.decode(), rtol, atol, path, out)
        except UnicodeDecodeError:
            out.append(f"{path}: binary file differs")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args()
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", "Tests", "Sources"],
                            capture_output=True, text=True, check=True).stdout.splitlines()
    noise, known, differs = [], [], []
    expected = set(listed("known-differences.txt"))
    for line in status:
        code, path = line[:2], line[3:]
        if path in expected:
            known.append(path)
        elif code == "??":
            differs.append(f"{path}: written by a generator but not committed")
        elif "D" in code:
            differs.append(f"{path}: committed but no longer written")
        else:
            found = judge(path, args.rtol, args.atol)
            (differs.extend(found) if found else noise.append(path))
    for path in noise:
        print(f"float noise  {path}")
    for path in known:
        print(f"known        {path} (tools/pins/known-differences.txt)")
    for line in differs:
        print(f"DIFFERS      {line}")
    print(f"{len(status)} files changed: {len(noise)} within float noise (rtol {args.rtol:g}, atol {args.atol:g}), "
          f"{len(known)} known, {len(status) - len(noise) - len(known)} differ")
    return 1 if differs else 0


if __name__ == "__main__":
    sys.exit(main())

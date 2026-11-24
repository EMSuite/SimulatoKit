"""SimulatoKit image Gaussian filter: Images.gaussian_filter (abTEM 1.0.10, scipy.ndimage.gaussian_filter underneath).

Prints the numbers that Tests/SimulatoCoreTests/ImageGaussianFilterTests.swift holds inline (no fixture file): a planted
7 x 9 image at sampling (0.3, 0.25) A, values sin(0.7 i) + cos(0.4 j)^2 + 0.05 i j + 2, filtered with sigma in A for each
boundary (periodic, reflect, constant), sigma per axis, on one axis only, and a kernel wider than the image. Each result is
re-derived from the published Gaussian kernel (scipy's: truncate 4, radius int(4 sigma + 0.5), normalised, correlated along
axis 0 then axis 1) and asserted equal. abTEM runs in float64 (precision "float64"); the test holds the values at rtol 1e-7
(assert_allclose defaults; neither abTEM nor scipy states a bound for this call).

Run from the repo root:  .venv/bin/python -I -B tools/pins/image-gaussian/pins.py
"""

import numpy as np
import abtem
from abtem.measurements import Images

abtem.config.set({"device": "cpu", "precision": "float64"})

i, j = np.meshgrid(np.arange(7), np.arange(9), indexing="ij")
arr = np.sin(0.7 * i) + np.cos(0.4 * j) ** 2 + 0.05 * i * j + 2
img = Images(arr, sampling=(0.3, 0.25))


def text(a, sigma_px, mode, cval):
    out = a.astype(float)
    for axis, s in enumerate(sigma_px):
        if s <= 1e-15:
            continue
        r = int(4.0 * s + 0.5)
        x = np.arange(-r, r + 1)
        w = np.exp(-0.5 / (s * s) * x ** 2)
        w /= w.sum()
        n = out.shape[axis]
        res = np.zeros_like(out)
        for k, wk in zip(x, w):
            idx = np.arange(n) + k
            if mode == "wrap":
                src = np.take(out, idx % n, axis=axis)
            elif mode == "reflect":
                m = idx % (2 * n)
                m = np.where(m < n, m, 2 * n - 1 - m)
                src = np.take(out, m, axis=axis)
            else:
                inside = (idx >= 0) & (idx < n)
                src = np.take(out, np.clip(idx, 0, n - 1), axis=axis)
                shape = [1, 1]
                shape[axis] = n
                src = np.where(inside.reshape(shape), src, cval)
            res += wk * src
        out = res
    return out


CASES = [
    ("periodic 0.4", 0.4, "periodic", 0.0),
    ("reflect 0.5 0.2", (0.5, 0.2), "reflect", 0.0),
    ("constant 0.6 cval 1.5", 0.6, "constant", 1.5),
    ("periodic 0.6 0", (0.6, 0.0), "periodic", 0.0),
    ("periodic 3.0", 3.0, "periodic", 0.0),
    ("reflect 3.0", 3.0, "reflect", 0.0),
    ("constant 3.0 cval 0", 3.0, "constant", 0.0),
]

for name, sigma, boundary, cval in CASES:
    out = img.gaussian_filter(sigma, boundary=boundary, cval=cval).array
    s = (sigma, sigma) if np.isscalar(sigma) else sigma
    px = tuple(a / d for a, d in zip(s, img.sampling))
    mode = {"periodic": "wrap"}.get(boundary, boundary)
    ref = text(arr, px, mode, cval)
    err = float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))
    assert err < 1e-13, (name, err)
    print(name, "values", out.ravel().tolist())

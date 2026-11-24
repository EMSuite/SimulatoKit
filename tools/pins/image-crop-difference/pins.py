"""SimulatoKit image crop and relative difference: Images.crop and BaseMeasurements.relative_difference (abTEM 1.0.10).

Prints the numbers that Tests/SimulatoCoreTests/ImageComparisonTests.swift holds inline (no fixture file): a planted 7 x 9
image a at sampling (0.3, 0.25) A, values sin(0.7 i) + cos(0.4 j)^2 + 0.05 i j + 2, cropped by five windows (one rounding
half to even on both axes, one cut at the far edge, the whole image, one centred), and its relative difference to
b = 1.01 a + 0.02 cos(i), with min_relative_tol 0 and, on a copy of a with two pixels set to 0 and -0.5, 0.2. A crop is a
slice, checked exact against NumPy's own slice; the differences are re-derived as 100 (a - b) / a and asserted equal.
abTEM runs in float64; the test holds the differences at rtol 1e-7 (assert_allclose defaults; abTEM states no bound).

Run from the repo root:  .venv/bin/python -I -B tools/pins/image-crop-difference/pins.py
"""

import numpy as np
import abtem
from abtem.measurements import Images

abtem.config.set({"device": "cpu", "precision": "float64"})

i, j = np.meshgrid(np.arange(7), np.arange(9), indexing="ij")
arr = np.sin(0.7 * i) + np.cos(0.4 * j) ** 2 + 0.05 * i * j + 2
img = Images(arr, sampling=(0.3, 0.25))

CROPS = [
    ("inside", (1.2, 1.0), (0.3, 0.5), False),
    ("half to even", (0.75, 0.625), (0.45, 0.125), False),
    ("cut at the edge", (1.2, 1.0), (1.8, 1.75), False),
    ("whole", (2.1, 2.25), (0.0, 0.0), False),
    ("centred", (1.0, 1.0), (0.0, 0.0), True),
]

for name, extent, offset, centered in CROPS:
    c = img.crop(extent, offset, centered=centered)
    if centered:
        offset = (img.extent[0] / 2 - extent[0] / 2, img.extent[1] / 2 - extent[1] / 2)
    x0 = int(np.round(7 * offset[0] / img.extent[0]))
    y0 = int(np.round(9 * offset[1] / img.extent[1]))
    mx = int(np.round(7 * extent[0] / img.extent[0]))
    my = int(np.round(9 * extent[1] / img.extent[1]))
    assert np.array_equal(c.array, arr[x0:x0 + mx, y0:y0 + my]), name
    assert c.sampling == img.sampling, name
    print(name, "origin", (x0, y0), "shape", c.array.shape)

other = arr * 1.01 + 0.02 * np.cos(i)
b = Images(other, sampling=(0.3, 0.25))
d = img.relative_difference(b).array
assert np.array_equal(d, ((arr - other) / arr) * 100)
print("difference tol 0", d.ravel().tolist())

z = arr.copy()
z[1, 1] = 0.0
z[2, 2] = -0.5
d2 = Images(z, sampling=(0.3, 0.25)).relative_difference(b, min_relative_tol=0.2).array
ref = np.where(np.abs(z) >= 0.2 * z.max(), ((z - other) / np.where(z == 0, 1, z)) * 100, np.nan)
assert np.array_equal(d2, ref, equal_nan=True)
print("difference tol 0.2 NaN at", np.argwhere(np.isnan(d2)).tolist())
print("difference tol 0.2", np.nan_to_num(d2, nan=-1e300).ravel().tolist())

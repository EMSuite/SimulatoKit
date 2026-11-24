# Changelog

All notable changes to SimulatoKit, newest first. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [semantic versioning](https://semver.org) (0.x: a minor version may break the API). Each release says
what a consumer must change under **Migration**. Every version below is a git tag; the date is the tag's. From the next release on,
release-please writes each entry from the merged pull requests' titles, and a breaking change's Migration is its
`BREAKING CHANGE:` note.

## [0.7.1](https://github.com/EMSuite/SimulatoKit/compare/0.7.0...0.7.1) (2026-11-24)


### Bug Fixes

* the azimuthal average of a tiny radial sampling no longer traps ([#132](https://github.com/EMSuite/SimulatoKit/issues/132)) ([182f4ba](https://github.com/EMSuite/SimulatoKit/commit/182f4ba192fcfd8bcc2db78ffc1490ec50f3a171))

## [0.7.0](https://github.com/EMSuite/SimulatoKit/compare/0.6.0...0.7.0) (2026-11-24)


### ⚠ BREAKING CHANGES

* `SimulationProgress.Stage.buildingSlab` and `SimulationProgress.Unit.configurations` are removed; no run reported them. Migration: remove `case .buildingSlab` and `case .configurations` from switches over `SimulationProgress.Stage` and `SimulationProgress.Unit`.

### Features

* atom-model specimens and slabs (`CrystalCell(cellRows:...)`, `CrystalCell.padded`, `SlabBuilder.slab(ofModel:)`); the unused progress cases are gone ([#135](https://github.com/EMSuite/SimulatoKit/issues/135)) ([fa3637c](https://github.com/EMSuite/SimulatoKit/commit/fa3637c79cec4d7a3c231fc5271ffc59001de184))
* flexible annular detector (radial bins per probe position) ([#124](https://github.com/EMSuite/SimulatoKit/issues/124)) ([e5e0db4](https://github.com/EMSuite/SimulatoKit/commit/e5e0db47877aeb896c22fa876d3ba9a17306c501))
* partly occupied sites drawn per configuration ([#123](https://github.com/EMSuite/SimulatoKit/issues/123)) ([e566844](https://github.com/EMSuite/SimulatoKit/commit/e56684434f680e9517bbb64736b3b65719002a3e))
* repeating slices share their transmission functions; Stop within one slice; faster CPU and Metal multislice ([#138](https://github.com/EMSuite/SimulatoKit/issues/138)) ([10384ad](https://github.com/EMSuite/SimulatoKit/commit/10384ad475a79edaf3c231a3ff4afa814f18b754))


### Bug Fixes

* a slab's z is depth below the face nearest the source ([#136](https://github.com/EMSuite/SimulatoKit/issues/136)) ([6666782](https://github.com/EMSuite/SimulatoKit/commit/666678232b2e643371841ce0b588fefb272e128b))


### Performance Improvements

* the slab builder cuts atoms only for the rung it takes, and Stop reaches it ([#135](https://github.com/EMSuite/SimulatoKit/issues/135)) ([fa3637c](https://github.com/EMSuite/SimulatoKit/commit/fa3637c79cec4d7a3c231fc5271ffc59001de184))

## [0.6.0](https://github.com/EMSuite/SimulatoKit/compare/0.5.0...0.6.0) (2026-11-24)


### ⚠ BREAKING CHANGES

* initialisers throw invalidShape instead of trapping on caller data ([#82](https://github.com/EMSuite/SimulatoKit/issues/82))
* Kirkland parameterisation, chosen by Parameterization ([#79](https://github.com/EMSuite/SimulatoKit/issues/79))

### Features

* a crystal cell and its orthogonal slab along a zone axis ([#73](https://github.com/EMSuite/SimulatoKit/issues/73)) ([9cb3cfc](https://github.com/EMSuite/SimulatoKit/commit/9cb3cfcbd6274ead57418c07f7a8a1f0b0155c56))
* a Gaussian filter on images, as abTEM's Images.gaussian_filter ([#106](https://github.com/EMSuite/SimulatoKit/issues/106)) ([b26bf50](https://github.com/EMSuite/SimulatoKit/commit/b26bf501b7758b66175f69fb1cc7a06c1b0d41c1)), closes [#105](https://github.com/EMSuite/SimulatoKit/issues/105)
* an HRTEM image series over defocus and thickness from stored exit waves ([#90](https://github.com/EMSuite/SimulatoKit/issues/90)) ([407cf77](https://github.com/EMSuite/SimulatoKit/commit/407cf7797870b556b7903635b9f7ec725e354652))
* beam intensities by hkl from the exit wave of a slab, at every stored thickness ([#98](https://github.com/EMSuite/SimulatoKit/issues/98)) ([65f93cf](https://github.com/EMSuite/SimulatoKit/commit/65f93cf7d281ac157f2c4ab0b3ec65285c2ca584))
* crop an image and its relative difference to another, as abTEM's Images.crop and relative_difference ([#110](https://github.com/EMSuite/SimulatoKit/issues/110)) ([72e0f93](https://github.com/EMSuite/SimulatoKit/commit/72e0f935eb002522bcc748fd2b94279be85ab9e9))
* initialisers throw invalidShape instead of trapping on caller data ([#82](https://github.com/EMSuite/SimulatoKit/issues/82)) ([dd2c17d](https://github.com/EMSuite/SimulatoKit/commit/dd2c17d3f6d5068898c802bbe8901c490fa95eb4))
* Kirkland parameterisation, chosen by Parameterization ([#79](https://github.com/EMSuite/SimulatoKit/issues/79)) ([abf6fb5](https://github.com/EMSuite/SimulatoKit/commit/abf6fb52ff882f6d7126d68295dbddaa32a5202b))
* plan a simulation's sampling, grid, memory and time before it runs ([#93](https://github.com/EMSuite/SimulatoKit/issues/93)) ([3c4c99c](https://github.com/EMSuite/SimulatoKit/commit/3c4c99cb7b57ff75d698935a5968790f6610b9b8))
* progress and cancellation in multislice and scans ([#77](https://github.com/EMSuite/SimulatoKit/issues/77)) ([0e678b4](https://github.com/EMSuite/SimulatoKit/commit/0e678b4b5c5a1ae1ab94e80f5a35ca82fd33c8b6))
* the HRTEM image series on Metal ([#104](https://github.com/EMSuite/SimulatoKit/issues/104)) ([b224ac2](https://github.com/EMSuite/SimulatoKit/commit/b224ac239637d841f00fb7aa148aecfa5a23b0ad))
* the simulation plan's memory counts the potential, the scan and a cold run's workspace ([#101](https://github.com/EMSuite/SimulatoKit/issues/101)) ([dea983d](https://github.com/EMSuite/SimulatoKit/commit/dea983d389d5e4f686f5768736d43f8b7427cbf5))
* thermal vibration amplitude per site, with the B and σ conversions ([#96](https://github.com/EMSuite/SimulatoKit/issues/96)) ([8f45df8](https://github.com/EMSuite/SimulatoKit/commit/8f45df87906e8df2f3d4cf795d4cb27bf5672364))
* transfer function profiles and the CTF's own Scherzer defocus, point resolution and crossover angle ([#103](https://github.com/EMSuite/SimulatoKit/issues/103)) ([2013c94](https://github.com/EMSuite/SimulatoKit/commit/2013c945ab9ec05b32358f56941a35e7937c6e77)), closes [#102](https://github.com/EMSuite/SimulatoKit/issues/102)


### Bug Fixes

* a custom scan pattern with negative axes is refused before the scan, not a trap after it ([#131](https://github.com/EMSuite/SimulatoKit/issues/131)) ([a2eaedf](https://github.com/EMSuite/SimulatoKit/commit/a2eaedfda1040fd2bfe4ebc72ca0052234b176e4))
* a mixed-energy batch is refused and a CTF keeps the wave's tilt ([#78](https://github.com/EMSuite/SimulatoKit/issues/78)) ([883364f](https://github.com/EMSuite/SimulatoKit/commit/883364fafa4752b226612aec61ad0352317e39a0))
* a STEM scan too large for memory is refused before anything is allocated ([#85](https://github.com/EMSuite/SimulatoKit/issues/85)) ([1a4fa64](https://github.com/EMSuite/SimulatoKit/commit/1a4fa6421ba0acbe28fee674666e339270ac9d66))
* build on Xcode 27, where Foundation's ProgressReporter clashes with the kit's ([#99](https://github.com/EMSuite/SimulatoKit/issues/99)) ([0b7852f](https://github.com/EMSuite/SimulatoKit/commit/0b7852f4a4a5806e980fac7a2e43de2b556a8f84))
* refuse values that hung or trapped Poisson noise, source size, detectors and Metal ([#81](https://github.com/EMSuite/SimulatoKit/issues/81)) ([0e6b1c2](https://github.com/EMSuite/SimulatoKit/commit/0e6b1c2a62b3eb4724a21946b1a5430440220713))


### Performance Improvements

* chunked scans prepare each potential once and reuse FFT plans and propagators ([#83](https://github.com/EMSuite/SimulatoKit/issues/83)) ([29459a1](https://github.com/EMSuite/SimulatoKit/commit/29459a11eda542b127b5b0b0a18df83ca9cc8707))
* scans compute the probe's aperture and aberration phase once, not per position ([#84](https://github.com/EMSuite/SimulatoKit/issues/84)) ([4931544](https://github.com/EMSuite/SimulatoKit/commit/493154419591f8c1103c4d1cbfd9a5d554779b91))
* the azimuthal average adds each pixel to the rings near its radius only ([#113](https://github.com/EMSuite/SimulatoKit/issues/113)) ([62f6e24](https://github.com/EMSuite/SimulatoKit/commit/62f6e24263b742030cf6beec3fef51359d957ea9))
* the CPU potential sorts its atoms into slices and elements in one pass ([#86](https://github.com/EMSuite/SimulatoKit/issues/86)) ([da9d6dd](https://github.com/EMSuite/SimulatoKit/commit/da9d6dd2a285425e1d4df1ff4616dc4d0b22e43e))


### Fixtures and tests

* reference pins for a CBED thickness series ([#112](https://github.com/EMSuite/SimulatoKit/issues/112)) ([2e58364](https://github.com/EMSuite/SimulatoKit/commit/2e58364f99486eb932995d93b8678d598834daa5))
* reference pins for a specimen from an atom model ([#118](https://github.com/EMSuite/SimulatoKit/issues/118)) ([59079dc](https://github.com/EMSuite/SimulatoKit/commit/59079dc6e837ee75574e4b94acc28a51fcc3f7d5))
* reference pins for the flexible annular detector ([#109](https://github.com/EMSuite/SimulatoKit/issues/109)) ([a2902f1](https://github.com/EMSuite/SimulatoKit/commit/a2902f186acd00cf499333cc1f2cac409aa30517))
* reference pins for the Si [110] HRTEM image chain ([#115](https://github.com/EMSuite/SimulatoKit/issues/115)) ([47587b0](https://github.com/EMSuite/SimulatoKit/commit/47587b04cc6ef9c0e454164daf8ba9ce166b9a1c))

## [0.5.0](https://github.com/mac4DSTEM/SimulatoKit/compare/0.4.0...0.5.0) (2026-10-31)


### Features

* azimuthal average of a centred diffraction pattern ([#39](https://github.com/mac4DSTEM/SimulatoKit/issues/39)) ([82ec3d0](https://github.com/mac4DSTEM/SimulatoKit/commit/82ec3d0bf0bc8fcf358b15c2484e7a092576435f))
* Fourier-space image interpolation and integrateDisc(interpolate:) ([#41](https://github.com/mac4DSTEM/SimulatoKit/issues/41)) ([f39aa4b](https://github.com/mac4DSTEM/SimulatoKit/commit/f39aa4b71ca2afbe0380732c11c537a77f4fe485))
* LineScan matched to a probe: end=None and Nyquist sampling defaults ([#40](https://github.com/mac4DSTEM/SimulatoKit/issues/40)) ([17346f9](https://github.com/mac4DSTEM/SimulatoKit/commit/17346f9c0cdf4097c90f282d1e017a6265eeb89b))
* Poisson noise samples and frozen-phonon exit waves ([#33](https://github.com/mac4DSTEM/SimulatoKit/issues/33)) ([a40af46](https://github.com/mac4DSTEM/SimulatoKit/commit/a40af46aad8652618f0919ad750a566e69d2ea16))
* ProjectedPotential.project and tiled, Potential.project ([#38](https://github.com/mac4DSTEM/SimulatoKit/issues/38)) ([78a6cd2](https://github.com/mac4DSTEM/SimulatoKit/commit/78a6cd250a4124d82abf90d5e0eff6390fc8e0ce))
* segmented-detector DPC: integrate, integrateRadial and differentials on a polar scan ([#34](https://github.com/mac4DSTEM/SimulatoKit/issues/34)) ([aca04b2](https://github.com/mac4DSTEM/SimulatoKit/commit/aca04b2b1c1b694819b2f85c7d599fd98691f0bb))
* Slicing.crystalSliceThicknesses, slices that follow the crystal planes ([#37](https://github.com/mac4DSTEM/SimulatoKit/issues/37)) ([4447588](https://github.com/mac4DSTEM/SimulatoKit/commit/444758803f978f6f472cc87991aeb939568dbea6))


### Performance Improvements

* generic odd-radix FFT pass for lengths with prime factors above 7 ([#45](https://github.com/mac4DSTEM/SimulatoKit/issues/45)) ([1f626b6](https://github.com/mac4DSTEM/SimulatoKit/commit/1f626b6ccee898176b7672f77efcf9f0ad9c8f67))
* mixed-radix FFT for 2, 3, 5, 7 lengths off vDSP (250² 2.2x, 500² 2.3x) ([0fbd053](https://github.com/mac4DSTEM/SimulatoKit/commit/0fbd053e73fde33ab4398cd3e0f735a4f8aa2c7b))
* mixed-radix FFT for lengths off vDSP that factor into 2, 3, 5 and 7 ([#42](https://github.com/mac4DSTEM/SimulatoKit/issues/42)) ([0fbd053](https://github.com/mac4DSTEM/SimulatoKit/commit/0fbd053e73fde33ab4398cd3e0f735a4f8aa2c7b))

## [0.4.0](https://github.com/mac4DSTEM/SimulatoKit/compare/0.3.1...0.4.0) (2026-10-22)

The study release: line scans and their profiles, integrated centre of mass, and exit waves per scan position. Pinned against abTEM 1.0.10 float64; `docs/parity.md` goes to 37 present, 36 partial, 55 missing of 128 rows (from 28, 36, 59 of 123).

### Added
- `LineScan(start:end:gpts:sampling:endpoint:fractional:extent:)` with `addingMargin` and `atPosition` (abTEM `LineScan`, `.add_margin`, `.at_position`); `STEM.scan` takes any `ScanPattern` (`GridScan` or `LineScan`) and returns `scanShape` `[n]` for a line.
- `RealSpaceLineProfiles` and `ReciprocalSpaceLineProfiles` with `width(height:)` and `interpolate(sampling:gpts:order:endpoint:)`; `ScanMeasurement.lineProfiles(for:)` turns a line-scan measurement into profiles.
- `RealField.interpolateLine`, `interpolateLineAtPosition` and, for centred diffraction patterns, `interpolateReciprocalLine`, `interpolateReciprocalLineAtPosition` (abTEM `interpolate_line`, `interpolate_line_at_position`: spline orders 0 to 5, width, margin, fractional coordinates).
- `RealField.integratingGradient(x:y:)` and `ComplexField.integratedGradient(samplingX:samplingY:)`: abTEM's `Images.integrate_gradient`, the Fourier-space integral of a two-component gradient field (NumPy `fftfreq`, zero frequency set to 0, no padding, minimum shifted to 0), CPU float64 on the kit's FFT, any length from 2.
- `ScanMeasurement.integratedCenterOfMass(samplingX:samplingY:)` and `gradientImages(...)`: the integrated centre of mass (iCOM / DPC) of a `CenterOfMassDetector` scan, as abTEM's `integrated_center_of_mass`.
- `RealField.integrateDisc(x:y:radius:mean:border:)` (abTEM `integrate_disc`) and `RealField.tiled(repetitionsX:repetitionsY:)` (`Images.tile`).
- `ImageOperationError`.
- `WavesDetector` (abTEM's `WavesDetector(gpts)`): `STEM.scan(probe:potentials:scan:detector:backend:options:memoryLimit:)`
  returns a `ScanWaves`, the complex exit wave at every scan position (`[position][row][column]`, subscript `scanWaves[ix, iy]`
  gives a `Waves`). `WavesDetector(rows:columns:)` resamples each wave in Fourier space (crop or zero-pad over the same
  extent), as abTEM's `fft_interpolate`. Pinned against abTEM 1.0.10 float64 at 1.8e-15 of the wave maximum (assert_allclose
  rtol 1e-7, atol 1e-12 of the maximum); `.metal` (float32) holds `MetalAccuracy.registered.wave` (measured 3.6e-7).
- `Waves.resampled(rows:columns:)` (SimulatoCore): abTEM's `fft_interpolate` for a wave, odd and even sizes, crop and pad.
- `WavesDetectorError`: `tooLarge` (positions x rows x columns x 16 bytes above half of the physical memory, or the
  `memoryLimit` given) and `severalPotentials` (frozen phonons).

### Migration

`STEM.scan(scan:)` is now generic over `ScanPattern`; a caller passing a `GridScan` needs no change. `ScanMeasurement.scanShape` can have one entry for a line scan: code that indexes `scanShape[1]` must check `scanShape.count` first, and the `(ix, iy)` subscript and `image(...)` need two scan axes. `STEM.scan` with a `WavesDetector` returns `ScanWaves`, a different type from the measurement of the other detectors (one potential only: a frozen-phonon list is refused with `WavesDetectorError.severalPotentials`; scan each configuration alone). A review of the new surface turned inputs that trapped into throws before the tag: `SimulatoError.gridTooLarge(rows:columns:required:limit:)` is a new case (a consumer that switches exhaustively over `SimulatoError` adds it; `Waves.resampled` throws it above half of the physical memory), `LineScan` caps explicit `gpts` below 1e9, `LineProfiles.interpolate` caps its points and the profile count times points at 2^31, `RealField.tiled` caps the result at 2^28 elements, `integrateDisc` refuses a disc or position beyond what an `Int` holds, and `ScanMeasurement.integratedCenterOfMass` and `gradientImages` refuse a one-axis scan. Everything else is additive.

## [0.3.1](https://github.com/mac4DSTEM/SimulatoKit/compare/0.3.0...0.3.1) (2026-10-21)

Documentation only; no source change and no number moves.

- The 0.3.0 entry no longer claims Bluestein runs at NumPy's speed; it is 2x to 7x slower (`docs/open-items.md` 1), as measured.
- README: install line `from: "0.3.1"` (was `"0.1.0"`); the "no noise" sentence now names `ScanMeasurement.poissonNoise`.
- The 0.3.0 Migration paragraph now lists the defaulted `tilt:` argument on `Probe.init`, `Waves.planeWave` and `Multislice.propagator`, not only `Waves.init`.
- Added `docs/status.md`.

### Migration

None.

## [0.3.0](https://github.com/mac4DSTEM/SimulatoKit/compare/0.2.1...0.3.0) (2026-10-21)

- Any grid size: `Grid`, `Potential` and every FFT accept any length of at least 2, on `.cpu` and `.metal` (before:
  f · 2^m with f in 1, 3, 5, 15 and m ≥ 3). vDSP's DFT runs where it has the length, Bluestein's algorithm (own
  implementation; ImagoKit's FFT is not a dependency, the kits stay standalone) elsewhere. Bluestein is slower than NumPy's
  mixed-radix FFT: 2x to 7x on the sizes measured (3.8 ms against 0.57 ms at 250², `docs/open-items.md` 1); the numbers are right, only speed is open. Added `Grid.hasDirectTransform(_:)`. A 90 x 75 grid is pinned against abTEM (rtol 1e−7, plane wave and
  probe).
- Added `Potential(structure:sampling:sliceThickness:)` (a sampling or an (x, y) pair), `Grid(extentX:extentY:samplingX:samplingY:)`
  and `GridScan(start:end:samplingX:samplingY:endpoint:)`: ⌈extent / sampling⌉ points (one more with `endpoint`), as abTEM.
- Added beam tilt [mrad]: `Waves.tilt`, `Waves.planeWave(…tilt:)`, `Probe(…tilt:)` and `Multislice.propagator(…tilt:)`. As in
  abTEM it enters the Fresnel propagator only; exit waves keep it. Pinned against abTEM (plane wave and probe); `.metal`
  matches `.cpu` at the registered bound. A `.metal` batch must share one tilt (`SimulatoError.invalidTilt`).
- Added `CenterOfMassDetector`, `ScanMeasurement.gaussianSourceSize(…)`, `ScanMeasurement.poissonNoise(…)` and a public
  `ScanMeasurement.init`. The Poisson draw uses the kit's generator, not NumPy's stream.
- `docs/parity.md` rewritten from abTEM's source (28 present, 36 partial, 59 missing of 123 rows, each marked study, useful or
  not planned); `docs/benchmarks.md` and `tools/benchmarks/` added; `docs/open-items.md` added.
- Added `SimulatoError.invalidSampling` and `.invalidTilt`; `.unsupportedGridLength` is now thrown only for a length below 2.
- No number moves for a grid or call that worked in 0.2.1.

### Migration

Nothing is required. `SimulatoError` gained two cases, so an exhaustive `switch` over it needs a `default` or the
two new cases. `Grid.isSupportedLength(_:)` is now true for every n ≥ 2: code that used it to pick vDSP lengths should call
`Grid.hasDirectTransform(_:)`. The `Waves` initialiser has a new defaulted `tilt:` argument, so a type that mirrors its
signature must add it. The same defaulted `tilt:` (zero) is new on `Probe.init`, `Waves.planeWave` and `Multislice.propagator`: a call
that worked in 0.2.1 compiles and gives the same numbers, and a type that mirrors any of those signatures must add it.

## [0.2.1](https://github.com/mac4DSTEM/SimulatoKit/compare/0.2.0...0.2.1) (2026-10-15)

- Added `MetalAccuracy.detectorFloor` (1e-5) and `MetalAccuracy.detectorBound(imageMaximum:incidentIntensity:)`: the
  bound for Metal detector images is 2e-4 x max(the image's maximum, 1e-5 x the incident intensity).
- No number moves.

### Migration

None.

## [0.2.0](https://github.com/mac4DSTEM/SimulatoKit/compare/0.1.0...0.2.0) (2026-10-15)

- Every choice the kit makes is now a parameter, and every default is the previous value. On the defaults no number
  moves: 14 bit digests of both backends and both potential forms are identical on 0.1.0 and 0.2.0, and every
  fixture is unchanged.
- Added `ExecutionOptions` (`options:` on `Multislice.run`, `Multislice.runBatch` and `STEM.scan`),
  `MetalAccuracy.registered`, `Grid.matches(_:relativeTolerance:)`, `maximumSliceCount:` on the slicing functions and
  both `Potential` initialisers, and the constants of `ContrastTransferFunction.scherzerDefocus`, `pointResolution`
  and `nyquistSampling` as parameters.
- Added a DocC catalogue with one article per module.
- Fixed an invalid `import a, b` line in the README quick start.
- Lowered the platform floor to macOS 15 and iOS 18 (was macOS 27). Numbers are validated on macOS only; no test runs
  on iOS yet.

### Migration

Nothing, unless you want another choice: call sites written against 0.1.0 compile unchanged. One
source break: an unapplied reference to a changed function no longer has its old type
(`grids.allSatisfy(grid.matches)` fails to type-check; write `grids.allSatisfy { grid.matches($0) }`), and a protocol
requirement that mirrors an old signature must add the new parameter.

## [0.1.0](https://github.com/mac4DSTEM/SimulatoKit/compare/0.0.1...0.1.0) (2026-10-15)

- The first API meant for use by other packages; added a README that stands on its own, `docs/parity.md` (abTEM's
  public API against SimulatoKit), a consumer-check package that uses the whole public API from outside, CI and
  the Swift Package Index configuration.
- `DiffractionIntensity.init(grid:beam:values:)` is now public.
- `STEM.scan`, `Multislice.run` and `Multislice.runBatch` accept either `[ProjectedPotential]` / `ProjectedPotential`
  or `[Potential]` / `Potential`.
- Rewrote the example executable as a first-ten-minutes walkthrough (a CIF in; an HRTEM image and a STEM HAADF scan
  out).

### Migration

`Measurement` is renamed `ScanMeasurement` (it collided with Foundation's `Measurement`): replace the
name. An empty array literal passed to the functions above now needs its type (`[] as [ProjectedPotential]`).

## [0.0.1](https://github.com/mac4DSTEM/SimulatoKit/tree/0.0.1) (2026-10-15)

- The first version: potentials (Lobato parameterisation, infinite projection), multislice on `.cpu` (float64) and
  `.metal` (float32), probes and aberrations, frozen phonons, scan grids, annular, pixelated and segmented detectors,
  the STEM driver, and the contrast transfer function, checked against abTEM 1.0.10 in float64.
- Added `Potential` (the sliced structure, built lazily; on `.metal` it is built on the GPU) and the Metal performance
  work. `.metal` now renormalises every step, so its numbers moved within their documented bounds (the total-intensity
  error per slice from about 9e-7 to at most 1.4e-7); `.cpu` numbers did not move.

Unverified: the notes for 0.0.1 and 0.1.0 describe changes relative to the earlier untagged `main`; the exact
commit of each version could not be confirmed without tags.

Consumer notes for 0.0.1 to 0.2.1 are archived.

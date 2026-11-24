# ``SimulatoCore``

The shared foundation of SimulatoKit: electron beams, lateral grids, structures and slabs cut from crystal cells, complex and real fields, waves and the antialias aperture.

## Overview

SimulatoCore holds the types every other SimulatoKit module builds on. All conventions (array layout, FFT ordering, normalisation, constants) follow abTEM 1.0.10. The module also contains the float64 CPU FFT that serves as the numerical reference for the Metal path, and image operations on ``RealField``: the Fourier-space integral of a gradient field (`integratingGradient`), `integrateDisc`, `tiled`, a window in Å (`cropped`, `croppedCentered`) and the relative difference to another image in percent (`relativeDifference`), for a simulated image beside a measured one. ``Waves`` can be resampled in Fourier space (`resampled(rows:columns:)`, abTEM's `fft_interpolate`).

Read <doc:GridsWavesConventions> before indexing into any array the library returns.

## Topics

### Essentials

- <doc:GridsWavesConventions>
- ``SimulatoCore/SimulatoCore``

### Beam and constants

- ``ElectronBeam``
- ``PhysicalConstants``

### Grids and fields

- ``Grid``
- ``ComplexField``
- ``RealField``

### Waves

- ``Waves``
- ``Waves/Normalization``
- ``Waves/MaxAngle``
- ``Antialias``

### Structures

- ``AtomicStructure``
- ``AtomicStructure/Atom``
- ``CrystalCell``
- ``CrystalCell/Site``
- ``SlabBuilder``
- ``OrthogonalUnit``
- ``Slab``

### Progress

- ``SimulationProgress``
- ``SimulationProgress/Stage``
- ``SimulationProgress/Unit``
- ``SimulationProgressHandler``

### Errors

- ``SimulatoError``
- ``ImageOperationError``

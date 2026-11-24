//
//  BeamIntensities.swift
//  Role: the dynamical intensity of each zone-axis beam of a slab's exit waves, named by the crystal's Miller indices, at
//        every stored thickness (abTEM 1.0.10 `DiffractionPatterns.index_diffraction_spots` with no radius: the value of
//        the one pixel a reflection falls in, not a disc). The pixel of a reflection comes from the slab's integer
//        supercell matrix V, so the indices are exact: the crystal's hkl sits at the box frequency V hkl, and a beam of
//        the zero-order Laue zone has (V hkl)_z = 0.
//

import SimulatoCore

/// The intensity of each beam of the zero-order Laue zone in a slab's exit waves, named by the crystal's hkl, at every
/// stored thickness: the kit side of a dynamical spot table with a thickness slider.
public struct BeamIntensities: Sendable, Hashable {
    /// One beam of the zero-order Laue zone.
    public struct Beam: Sendable, Hashable {
        /// The crystal's Miller indices, in the cell the slab was built from (h a* + k b* + l c*).
        public let hkl: SIMD3<Int>
        /// The beam's signed frequency index on the exit waves' grid: it sits at (index.x / extent x, index.y / extent y)
        /// in 1/Å, the zero frequency at (0, 0).
        public let index: SIMD2<Int>
        /// The spatial frequency [1/Å] in the slab's box frame.
        public let spatialFrequency: SIMD2<Double>
    }

    /// The beams, by |g| and then by hkl (h first).
    public let beams: [Beam]
    /// The thickness of each exit wave [Å]: 0 for the incident wave, else the sum of the slices up to and including its
    /// plane.
    public let thicknesses: [Double]
    /// `intensities[t][b]`: beam b at thickness t, the diffraction pattern's value at the beam's pixel (abTEM's
    /// `diffraction_patterns()`; for a plane wave a fraction of the incident beam).
    public let intensities: [[Double]]

    /// The beams of the zero-order Laue zone that the exit waves' cutoff pattern holds (abTEM's
    /// `diffraction_patterns(max_angle="cutoff")`, |g| at most its largest frequency, abTEM's default `g_max`), read in
    /// each wave of `exitWaves`.
    ///
    /// `exitWaves` are the waves of a run through the slab's slices (`Potential(structure: slab.structure, ...,
    /// sliceThicknesses: slab.sliceThicknesses)`) with `exitPlanes`, in plane order, as `Multislice.run` returns them.
    /// - Throws: `SimulatoError.invalidShape` when the wave count is not the plane count or the waves are on different
    ///   grids; `gridMismatch` when the grid's extent is not the slab's box; `invalidExitPlanes` as ``ExitPlanes``.
    public init(exitWaves: [Waves], of slab: Slab, exitPlanes: ExitPlanes = .last) throws {
        let planes = try exitPlanes.planes(sliceCount: slab.sliceThicknesses.count)
        guard exitWaves.count == planes.count, let first = exitWaves.first else {
            throw SimulatoError.invalidShape("\(planes.count) exit planes need \(planes.count) waves, got \(exitWaves.count)")
        }
        let grid = first.grid
        for w in exitWaves where !w.grid.matches(grid) {
            throw SimulatoError.invalidShape("the exit waves are on different grids")
        }
        let box = slab.structure.box
        guard abs(grid.extentX - box.x) <= 1e-9 * box.x, abs(grid.extentY - box.y) <= 1e-9 * box.y else {
            throw SimulatoError.gridMismatch
        }
        var thicknesses: [Double] = []
        thicknesses.reserveCapacity(planes.count)
        var sum = 0.0, done = 0
        for p in planes {
            while done <= p {
                sum += slab.sliceThicknesses[done]
                done += 1
            }
            thicknesses.append(p < 0 ? 0 : sum)
        }
        let beams = Self.beams(of: slab, on: grid)
        let fft = try FFT2D(nx: grid.nx, ny: grid.ny)
        var intensities: [[Double]] = []
        intensities.reserveCapacity(exitWaves.count)
        for w in exitWaves {
            var f = w.field
            if w.normalization == .values {
                let n = Double(grid.count)
                for i in f.real.indices {
                    f.real[i] /= n
                    f.imaginary[i] /= n
                }
            }
            fft.forward(&f)
            var row: [Double] = []
            row.reserveCapacity(beams.count)
            for beam in beams {
                let ix = Self.wrap(beam.index.x, grid.nx), iy = Self.wrap(beam.index.y, grid.ny)
                let re = f.real[ix * grid.ny + iy], im = f.imaginary[ix * grid.ny + iy]
                row.append(re * re + im * im)
            }
            intensities.append(row)
        }
        self.beams = beams
        self.thicknesses = thicknesses
        self.intensities = intensities
    }

    /// The position of `hkl` in ``beams``, nil when it is not among them.
    public func beamIndex(of hkl: SIMD3<Int>) -> Int? {
        beams.firstIndex { $0.hkl == hkl }
    }

    /// The intensity of `hkl` at every stored thickness, nil when it is not among the beams.
    public func intensities(of hkl: SIMD3<Int>) -> [Double]? {
        guard let b = beamIndex(of: hkl) else { return nil }
        return intensities.map { $0[b] }
    }

    // MARK: The integer map

    /// The signed frequency index on the slab's grid of the crystal reflection `hkl`: (r_x (V hkl)_x, r_y (V hkl)_y), r the
    /// slab's lateral repeats; nil when hkl is not in the zero-order Laue zone ((V hkl)_z ≠ 0).
    public static func frequencyIndex(of hkl: SIMD3<Int>, in slab: Slab) -> SIMD2<Int>? {
        let v = slab.unit.supercell
        let m = SIMD3(dot(v[0], hkl), dot(v[1], hkl), dot(v[2], hkl))
        guard m.z == 0 else { return nil }
        return SIMD2(slab.repeats.x * m.x, slab.repeats.y * m.y)
    }

    /// The crystal reflection of the zero-order Laue zone at the signed frequency index `index` of the slab's grid,
    /// hkl = V⁻¹ (index.x / r_x, index.y / r_y, 0) in exact integers; nil when no reflection of the crystal sits there.
    public static func hkl(atFrequencyIndex index: SIMD2<Int>, in slab: Slab) -> SIMD3<Int>? {
        let r = slab.repeats
        guard index.x % r.x == 0, index.y % r.y == 0 else { return nil }
        let ux = index.x / r.x, uy = index.y / r.y
        let v = slab.unit.supercell
        // V⁻¹ = [b × c | c × a | a × b] / det V for the rows a, b, c of V; the third column meets u_z = 0.
        let bc = cross(v[1], v[2]), ca = cross(v[2], v[0])
        let det = dot(v[0], bc)
        guard det != 0 else { return nil }
        let n = SIMD3(bc.x * ux + ca.x * uy, bc.y * ux + ca.y * uy, bc.z * ux + ca.z * uy)
        guard n.x % det == 0, n.y % det == 0, n.z % det == 0 else { return nil }
        let h = SIMD3(n.x / det, n.y / det, n.z / det)
        guard dot(v[0], h) == ux, dot(v[1], h) == uy, dot(v[2], h) == 0 else { return nil }
        return h
    }

    /// The zero-order beams of the cutoff pattern on `grid`, by |g| (rounded to 1e−9 1/Å) and then by hkl.
    static func beams(of slab: Slab, on grid: Grid) -> [Beam] {
        let size = Antialias.cutoffPatternSize(on: grid)
        let lx = grid.extentX, ly = grid.extentY
        let gMax = max(Double((size.nx - 1) / 2) / lx, Double((size.ny - 1) / 2) / ly)
        var found: [(key: Double, beam: Beam)] = []
        for px in -(size.nx / 2)...(size.nx / 2) {
            for py in -(size.ny / 2)...(size.ny / 2) {
                let gx = Double(px) / lx, gy = Double(py) / ly
                let g = (gx * gx + gy * gy).squareRoot()
                guard g <= gMax * (1 + 1e-12) else { continue }
                guard let reflection = Self.hkl(atFrequencyIndex: SIMD2(px, py), in: slab) else { continue }
                let key = (g * 1e9).rounded(.toNearestOrEven)
                let beam = Beam(hkl: reflection, index: SIMD2(px, py), spatialFrequency: SIMD2(gx, gy))
                found.append((key: key, beam: beam))
            }
        }
        found.sort { l, r in
            if l.key != r.key { return l.key < r.key }
            if l.beam.hkl.x != r.beam.hkl.x { return l.beam.hkl.x < r.beam.hkl.x }
            if l.beam.hkl.y != r.beam.hkl.y { return l.beam.hkl.y < r.beam.hkl.y }
            return l.beam.hkl.z < r.beam.hkl.z
        }
        return found.map(\.beam)
    }

    static func wrap(_ i: Int, _ n: Int) -> Int {
        let w = i % n
        return w < 0 ? w + n : w
    }

    static func dot(_ a: SIMD3<Int>, _ b: SIMD3<Int>) -> Int {
        a.x * b.x + a.y * b.y + a.z * b.z
    }

    static func cross(_ a: SIMD3<Int>, _ b: SIMD3<Int>) -> SIMD3<Int> {
        let x = a.y * b.z - a.z * b.y
        let y = a.z * b.x - a.x * b.z
        let z = a.x * b.y - a.y * b.x
        return SIMD3(x, y, z)
    }
}

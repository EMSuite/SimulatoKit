//
//  HardeningTests.swift
//  Role: inputs that used to trap late in SimulatoDetector (a line scan's iCOM, a custom scan pattern of the wrong shape,
//        overflowing scans, empty profiles, huge widths) now throw their typed errors before any work.
//

import Foundation
import Testing
import SimulatoCore
import SimulatoDetector
import SimulatoMultislice
import SimulatoProbe

struct BadScan: ScanPattern {
    var positions: [SIMD2<Double>]
    var shape: [Int]
}

@Suite("Detector hardening")
struct DetectorHardeningTests {
    func setup() throws -> (Probe, ProjectedPotential) {
        let grid = try Grid(nx: 16, ny: 16, extentX: 4, extentY: 4)
        let beam = try ElectronBeam(energy: 200e3)
        let probe = try Probe(beam: beam, grid: grid, aperture: Aperture(semiangleCutoff: 20))
        let potential = ProjectedPotential.unchecked(grid: grid, slices: [PotentialSlice(thickness: 1, values: [Double](repeating: 0, count: 256))])
        return (probe, potential)
    }

    @Test func centreOfMassOfALineScanIsRefused() throws {
        let m = ScanMeasurement.unchecked(scanShape: [4], valueShape: [2], values: [Double](repeating: 0, count: 8))
        #expect(throws: SimulatoError.self) { try m.gradientImages(samplingX: 1, samplingY: 1) }
        #expect(throws: SimulatoError.self) { try m.integratedCenterOfMass(samplingX: 1, samplingY: 1) }
    }

    @Test func malformedScanPatternIsRefusedBeforeTheSimulation() throws {
        let (probe, potential) = try setup()
        let p = [SIMD2<Double>(1, 1), SIMD2(2, 2), SIMD2(3, 3)]
        let detector = try AnnularDetector(inner: 0, outer: 10)
        for shape in [[2, 2], [], [1, 1, 3]] {
            let scan = BadScan(positions: p, shape: shape)
            #expect(throws: SimulatoError.self) {
                try STEM.scan(probe: probe, potentials: [potential], scan: scan, detectors: [detector])
            }
        }
    }

    @Test func overflowingWaveScanIsTyped() throws {
        let (probe, potential) = try setup()
        let scan = try GridScan(start: SIMD2(0, 0), end: SIMD2(4, 4), nx: 1 << 40, ny: 1 << 40)
        #expect(throws: WavesDetectorError.self) {
            try STEM.scan(probe: probe, potentials: [potential], scan: scan, detector: WavesDetector())
        }
    }

    @Test func emptyProfilesAreRefused() throws {
        let empty = RealSpaceLineProfiles.unchecked(count: 0, values: [], sampling: 1)
        #expect(throws: SimulatoError.self) { try empty.width() }
        #expect(throws: SimulatoError.self) { try empty.interpolate(gpts: 3) }
        let p = RealSpaceLineProfiles.unchecked(count: 4, values: [0, 1, 1, 0], sampling: 1)
        #expect(throws: SimulatoError.self) { try p.interpolate(gpts: 1 << 40) }
        #expect(try p.interpolate(gpts: 8).count == 8)
    }

    @Test func hugeWidthIsRefused() throws {
        let f = RealField.unchecked(nx: 8, ny: 8, values: [Double](repeating: 1, count: 64), samplingX: 1, samplingY: 1)
        #expect(throws: SimulatoError.self) { try f.interpolateLine(end: SIMD2(4, 0), width: 1e300) }
        #expect(throws: SimulatoError.self) { try f.interpolateLine(end: SIMD2(4, 0), width: 1e12) }
        #expect(try f.interpolateLine(end: SIMD2(4, 0), width: 3).count > 0)
    }

    @Test func aScanTooLargeForMemoryIsRefusedBeforeAnyAllocation() throws {
        let (probe, potential) = try setup()
        // 10^10 positions of a 16 x 16 pixelated pattern: about 2e13 bytes, refused before the positions are made
        // (they alone would be 160 GB).
        let huge = try GridScan(start: SIMD2(0, 0), end: SIMD2(4, 4), nx: 100_000, ny: 100_000)
        #expect(throws: SimulatoError.self) {
            try STEM.scan(probe: probe, potentials: [potential], scan: huge, detectors: [PixelatedDetector()])
        }
        // A shape whose product overflows is refused the same way.
        let overflowing = BadScan(positions: [], shape: [Int.max, 2])
        #expect(throws: SimulatoError.self) {
            try STEMRun(probe: probe, potentials: [potential], scan: overflowing, detectors: [AnnularDetector(inner: 0, outer: 10)])
        }
        // The limit is exact: 6 positions x (one annular value x 8 + 16 bytes) = 144 bytes.
        let small = try GridScan(start: SIMD2(0, 0), end: SIMD2(4, 4), nx: 3, ny: 2)
        let annular: [any Detector] = [try AnnularDetector(inner: 0, outer: 10)]
        try STEM.checkMemory(small, detectors: annular, probe: probe, limit: 144)
        #expect(throws: SimulatoError.self) {
            try STEM.checkMemory(small, detectors: annular, probe: probe, limit: 143)
        }
    }
}

/// A detector whose values disagree with its own shape.
struct MiscountingDetector: Detector {
    func shape(for grid: Grid, beam: ElectronBeam) throws -> [Int] { [] }
    func detect(_ intensity: DiffractionIntensity) throws -> [Double] { [1, 2] }
}

/// #59: inputs that trapped or hung now throw.
@Suite("Detector hardening: traps and hangs")
struct DetectorTrapTests {
    /// A NaN or infinite value made the Poisson draw loop forever; a mean beyond an Int trapped.
    @Test func poissonNoiseRefusesValuesWithNoDraw() throws {
        for bad in [Double.nan, .infinity, 1e300] {
            let m = ScanMeasurement.unchecked(scanShape: [2], valueShape: [], values: [1, bad])
            #expect(throws: SimulatoError.self) { try m.poissonNoise(dose: 10, seed: 1) }
            #expect(throws: SimulatoError.self) { try m.poissonNoise(dose: 10, samples: 2, seed: 1) }
        }
        // −∞ is not finite either: refused, not clipped to 0.
        let negative = ScanMeasurement.unchecked(scanShape: [2], valueShape: [], values: [1, -.infinity])
        #expect(throws: SimulatoError.self) { try negative.poissonNoise(dose: 10, seed: 1) }
        // Finite values still draw, and a dose of 0 gives zeros.
        let fine = ScanMeasurement.unchecked(scanShape: [2], valueShape: [], values: [0.5, -3])
        #expect(try fine.poissonNoise(dose: 0, seed: 1).values == [0, 0])
        #expect(try fine.poissonNoise(dose: 1e6, seed: 1).values[1] == 0)
    }

    /// A source size far above the scan's sampling made a kernel radius beyond an Int (a trap) or memory.
    @Test func hugeSourceSizeIsRefused() throws {
        let m = ScanMeasurement.unchecked(scanShape: [4, 4], valueShape: [], values: [Double](repeating: 1, count: 16))
        #expect(throws: SimulatoError.self) { try m.gaussianSourceSize(sigma: 1e12, samplingX: 1, samplingY: 1) }
        #expect(throws: SimulatoError.self) { try m.gaussianSourceSize(sigma: 1, samplingX: 1e-300, samplingY: 1) }
        #expect(try m.gaussianSourceSize(sigma: 2, samplingX: 1, samplingY: 1).values.count == 16)
    }

    /// An angular sampling far finer than the outer angle made the segmented detector's crop beyond an Int.
    @Test func segmentedDetectorOnAFineAngularSamplingDoesNotTrap() throws {
        let grid = try Grid(nx: 8, ny: 8, extentX: 1e300, extentY: 1e300)
        let beam = try ElectronBeam(energy: 200e3)
        let p = DiffractionIntensity.unchecked(grid: grid, beam: beam, values: [Double](repeating: 1, count: 64))
        let d = try SegmentedDetector(radialBins: 2, azimuthalBins: 4, inner: 0, outer: 10)
        #expect(try d.detect(p).count == 8)
    }

    /// A custom detector returning more values than its shape trapped the scan.
    @Test func miscountingDetectorIsRefused() throws {
        let grid = try Grid(nx: 16, ny: 16, extentX: 4, extentY: 4)
        let probe = try Probe(beam: ElectronBeam(energy: 200e3), grid: grid, aperture: Aperture(semiangleCutoff: 20))
        let potential = ProjectedPotential.unchecked(grid: grid, slices: [PotentialSlice(thickness: 1, values: [Double](repeating: 0, count: 256))])
        let scan = try GridScan(start: SIMD2(0, 0), end: SIMD2(4, 4), nx: 2, ny: 2)
        #expect(throws: SimulatoError.self) {
            try STEM.scan(probe: probe, potentials: [potential], scan: scan, detectors: [MiscountingDetector()])
        }
    }

    /// A configuration index outside the ensemble trapped.
    @Test func configurationOutsideTheEnsembleIsRefused() throws {
        let structure = try AtomicStructure(box: SIMD3(4, 4, 4), atoms: [.init(atomicNumber: 13, position: SIMD3(2, 2, 2))])
        let phonons = try FrozenPhonons(structure: structure, sigmas: [13: 0.1], configurations: 2, seed: 1)
        #expect(throws: SimulatoError.self) { try phonons.configuration(2) }
        #expect(throws: SimulatoError.self) { try phonons.configuration(-1) }
        #expect(try phonons.configuration(1).atoms.count == 1)
    }

    /// #80: the measurement, intensity, exit-wave and line-profile initialisers trapped on values that do not fill
    /// their shapes.
    @Test func measurementsRefuseValuesThatDoNotFillTheShape() throws {
        #expect(throws: SimulatoError.self) { try ScanMeasurement(scanShape: [2, 2], valueShape: [], values: [1, 2, 3]) }
        #expect(throws: SimulatoError.self) { try ScanMeasurement(scanShape: [], valueShape: [], values: [1]) }
        #expect(throws: SimulatoError.self) { try ScanMeasurement(scanShape: [1, 1, 1], valueShape: [], values: [1]) }
        #expect(throws: SimulatoError.self) { try ScanMeasurement(scanShape: [-1, -1], valueShape: [], values: [1]) }
        #expect(try ScanMeasurement(scanShape: [2], valueShape: [2], values: [1, 2, 3, 4]).values.count == 4)
        let grid = try Grid(nx: 4, ny: 4, extentX: 1, extentY: 1)
        let beam = try ElectronBeam(energy: 200e3)
        #expect(throws: SimulatoError.self) { try DiffractionIntensity(grid: grid, beam: beam, values: [1]) }
        let template = Waves.planeWave(beam: beam, grid: grid)
        #expect(throws: SimulatoError.self) { try ScanWaves(scanShape: [2], template: template, real: [], imaginary: []) }
        #expect(throws: SimulatoError.self) {
            try ScanWaves(scanShape: [1, 1], template: template, real: [Double](repeating: 0, count: 16), imaginary: [])
        }
        #expect(try ScanWaves(scanShape: [1, 1], template: template, real: [Double](repeating: 0, count: 16),
                              imaginary: [Double](repeating: 0, count: 16)).positionCount == 1)
        #expect(throws: SimulatoError.self) { try RealSpaceLineProfiles(count: 3, values: [1, 2], sampling: 1) }
        #expect(throws: SimulatoError.self) { try ReciprocalSpaceLineProfiles(ensembleShape: [2], count: 2, values: [1], sampling: 1) }
        #expect(try RealSpaceLineProfiles(ensembleShape: [2], count: 2, values: [1, 2, 3, 4], sampling: 1).count == 2)
    }
}

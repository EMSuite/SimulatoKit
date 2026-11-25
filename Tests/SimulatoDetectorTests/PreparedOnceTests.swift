//
//  PreparedOnceTests.swift
//  Role: scans run in chunks of positions prepare each potential once (#61): a WavesDetector scan builds its
//        transmission functions once, not once per chunk, a CPU STEM run makes its FFT plans once for every chunk, and
//        the numbers do not change (bit for bit against one chunk). The probe's aperture and aberration phase are
//        computed once per scan, not once per position, and with frozen phonons once for every configuration, as are the
//        FFT plans and propagators. A prepared batch refuses another energy or grid.
//

import Testing
import SimulatoCore
import SimulatoDetector
import SimulatoKitTestSupport
import SimulatoMultislice
import SimulatoProbe

@Suite("Scans prepare each potential once")
struct PreparedOnceTests {
    /// Al 1 x 1 x 2 on 32², 200 keV, 20 mrad, a 3 x 2 scan.
    static func setup() throws -> (probe: Probe, potential: Potential, scan: GridScan) {
        let potential = try Potential(structure: aluminiumStructure(cells: (1, 1, 2)), nx: 32, ny: 32, sliceThickness: 2.025)
        let probe = try Probe(beam: ElectronBeam(energy: 200e3), grid: potential.grid, aperture: Aperture(semiangleCutoff: 20))
        let scan = try GridScan(start: SIMD2(0, 0), end: SIMD2(4.05, 4.05), nx: 3, ny: 2)
        return (probe, potential, scan)
    }

    /// One position per chunk (and per CPU round).
    static let onePerChunk = ExecutionOptions(maximumCPUThreads: 2, scanChunkBytes: 1)

    static func counting<T>(_ body: () throws -> T) rethrows -> (T, WorkCounter) {
        let counter = WorkCounter()
        let value = try WorkCounter.$current.withValue(counter) { try body() }
        return (value, counter)
    }

    @Test func aWavesScanBuildsItsTransmissionFunctionsOnce() throws {
        let (probe, potential, scan) = try Self.setup()
        let built = try potential.build()
        let (chunked, counter) = try Self.counting {
            try STEM.scan(probe: probe, potentials: [built], scan: scan, detector: WavesDetector(), options: Self.onePerChunk)
        }
        // Six chunks of one position: one preparation and one probe spectrum (each was six).
        #expect(counter.preparationCount == 1)
        #expect(counter.probeSpectrumCount == 1)
        let (lazy, lazyCounter) = try Self.counting {
            try STEM.scan(probe: probe, potentials: [potential], scan: scan, detector: WavesDetector(), options: Self.onePerChunk)
        }
        #expect(lazyCounter.preparationCount == 1)
        let whole = try STEM.scan(probe: probe, potentials: [built], scan: scan, detector: WavesDetector())
        #expect(chunked.real == whole.real && chunked.imaginary == whole.imaginary)
        #expect(lazy.real == whole.real && lazy.imaginary == whole.imaginary)
    }

    @Test func frozenPhononWavesPrepareOncePerConfiguration() throws {
        let (probe, potential, scan) = try Self.setup()
        let (scans, counter) = try Self.counting {
            try STEM.scanFrozenPhonons(probe: probe, potentials: [potential, potential], scan: scan,
                                       detector: WavesDetector(), options: Self.onePerChunk)
        }
        #expect(scans.count == 2)
        #expect(counter.preparationCount == 2)
        // One probe spectrum and one FFT plan for both configurations (each was two).
        #expect(counter.probeSpectrumCount == 1)
        #expect(counter.fftPlanCount == 1)
        let single = try STEM.scan(probe: probe, potentials: [potential], scan: scan, detector: WavesDetector(),
                                   options: Self.onePerChunk)
        for waves in scans {
            #expect(waves.real == single.real && waves.imaginary == single.imaginary)
        }
    }

    @Test func aFrozenPhononRunMakesItsProbeAndPlansOnceForEveryConfiguration() throws {
        let (probe, potential, scan) = try Self.setup()
        let detectors: [any Detector] = [try AnnularDetector(inner: 0, outer: 20), PixelatedDetector()]
        let (twice, counter) = try Self.counting {
            try STEM.scan(probe: probe, potentials: [potential, potential], scan: scan, detectors: detectors,
                          options: Self.onePerChunk)
        }
        // Two configurations: two preparations, but one probe spectrum and one FFT plan (each was two).
        #expect(counter.preparationCount == 2)
        #expect(counter.probeSpectrumCount == 1)
        #expect(counter.fftPlanCount == 1)
        // Two equal configurations average to one, bit for bit ((a + a) / 2 = a).
        let once = try STEM.scan(probe: probe, potentials: [potential], scan: scan, detectors: detectors,
                                 options: Self.onePerChunk)
        #expect(twice.map(\.values) == once.map(\.values))
    }

    @Test(.enabled(if: Multislice.metalIsAvailable, "no Metal device"))
    func aFrozenPhononMetalRunMakesItsProbeSpectrumOnce() throws {
        let (probe, potential, scan) = try Self.setup()
        let (scans, counter) = try Self.counting {
            try STEM.scan(probe: probe, potentials: [potential, potential], scan: scan,
                          detectors: [try AnnularDetector(inner: 0, outer: 20)], backend: .metal)
        }
        #expect(scans.count == 1)
        #expect(counter.probeSpectrumCount == 1)
    }

    @Test func aCPURunMakesItsFFTPlansOnceForEveryChunk() throws {
        let (probe, potential, scan) = try Self.setup()
        let detectors: [any Detector] = [try AnnularDetector(inner: 0, outer: 20), PixelatedDetector()]
        let (chunked, counter) = try Self.counting {
            var run = try STEMRun(probe: probe, potentials: [potential], scan: scan, detectors: detectors,
                                  options: Self.onePerChunk)
            return try run.run()
        }
        // One position per chunk, so one worker: one plan for the six chunks, one preparation.
        #expect(counter.fftPlanCount == 1)
        #expect(counter.preparationCount == 1)
        #expect(counter.probeSpectrumCount == 1)
        let whole = try STEM.scan(probe: probe, potentials: [potential], scan: scan, detectors: detectors)
        #expect(chunked.map(\.values) == whole.map(\.values))
    }

    @Test func aPreparedBatchRefusesAnotherEnergyOrGrid() throws {
        let (probe, potential, _) = try Self.setup()
        let prepared = try Multislice.prepareBatch(potential, beam: probe.beam, tilt: .zero, backend: .cpu)
        let wave = try probe.build(at: SIMD2(1, 1))
        let exits = try Multislice.runBatch([wave], prepared: prepared)
        let reference = try Multislice.runBatch([wave], through: potential)
        #expect(exits[0].field == reference[0].field)
        let other = try Probe(beam: ElectronBeam(energy: 300e3), grid: potential.grid, aperture: Aperture(semiangleCutoff: 20))
        #expect(throws: SimulatoError.mixedEnergies(first: 200e3, other: 300e3)) {
            try Multislice.runBatch([other.build(at: SIMD2(1, 1))], prepared: prepared)
        }
        let small = try Probe(beam: probe.beam, grid: Grid(nx: 16, ny: 16, extentX: potential.grid.extentX,
                                                           extentY: potential.grid.extentY),
                              aperture: Aperture(semiangleCutoff: 20))
        #expect(throws: SimulatoError.gridMismatch) {
            try Multislice.runBatch([small.build(at: SIMD2(1, 1))], prepared: prepared)
        }
    }
}

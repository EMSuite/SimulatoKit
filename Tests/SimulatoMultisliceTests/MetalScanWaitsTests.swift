//
//  MetalScanWaitsTests.swift
//  Role: a Metal scan waits once per chunk of slices (#61): the probes and the intensities no longer wait on their own,
//        and the buffers reused across batches give the CPU's intensities, as one batch holding every position does.
//

import Testing
import SimulatoCore
import SimulatoMultislice
import SimulatoProbe
import SimulatoKitTestSupport

@Suite("Metal scan waits")
struct MetalScanWaitsTests {
    @Test(.enabled(if: Multislice.metalIsAvailable, "no Metal device"))
    func oneWaitPerChunkAndReusedBuffersChangeNoValue() throws {
        // Al, 2 x 2 x 2 cells on 64², 4 slices: one chunk per batch.
        let potential = try Potential(structure: aluminiumStructure(cells: (2, 2, 2)), nx: 64, ny: 64, sliceThickness: 2.025)
        let beam = try ElectronBeam(energy: 200e3)
        let probe = try Probe(beam: beam, grid: potential.grid, aperture: Aperture(semiangleCutoff: 20))
        let positions: [SIMD2<Double>] = (0..<5).map { SIMD2(0.7 * Double($0), 0.4 * Double($0)) }
        func scan(batchBytes: Int) throws -> (values: [[Double]], waits: Int) {
            var values: [[Double]] = []
            let counter = WorkCounter()
            try WorkCounter.$current.withValue(counter) { () throws -> Void in
                try Multislice.metalScanIntensities(
                    spectrum: probe.centredSpectrum(), positions: positions.count,
                    ramps: { (q: Int) throws -> (x: ComplexField, y: ComplexField) in try probe.shiftRamps(at: positions[q]) },
                    through: potential, beam: beam, options: ExecutionOptions(metalBatchBytes: batchBytes),
                    deliver: { (q: Int, v: [Double]) in
                        #expect(q == values.count, "positions in order")
                        values.append(v)
                    })
            }
            return (values, counter.deviceWaitCount)
        }
        // Batches of 2, 2 and 1 positions (a 64² wave costs 64 · 64 · 24 bytes), against one batch of all five.
        let small = try scan(batchBytes: 2 * 64 * 64 * 24)
        let whole = try scan(batchBytes: 512 << 20)
        #expect(small.waits == 3, "one wait per batch of one chunk, got \(small.waits)")
        #expect(whole.waits == 1)
        #expect(small.values.count == 5 && whole.values.count == 5)
        // Against the CPU: the same probes through the built potential, |FFT(ψ)|² (unnormalised, as the GPU's).
        let incidents = try positions.map { try probe.build(at: $0) }
        let built = try potential.build()
        let exits = try Multislice.runBatch(incidents, through: built)
        let fft = try FFT2D(nx: potential.grid.nx, ny: potential.grid.ny)
        for (q, exit) in exits.enumerated() {
            var f = exit.field
            fft.forward(&f)
            let cpu = f.squaredMagnitude()
            #expect(relMax(small.values[q], cpu) <= MetalAccuracy.registered.pattern, "position \(q), batches of 2")
            #expect(relMax(whole.values[q], cpu) <= MetalAccuracy.registered.pattern, "position \(q), one batch")
        }
    }
}

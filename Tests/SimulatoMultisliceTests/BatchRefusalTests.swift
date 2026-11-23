//
//  BatchRefusalTests.swift
//  Role: review findings of #56 (issues #60, #59): a batch of waves of different energies is refused on both backends (it
//        shared the first wave's transmission and propagator), and `MetalAccuracy` says which runs lie inside the range
//        its bounds were tested on; the Metal path's size checks (a tensor's bytes, the 32-bit scatter index).
//

import Testing
import SimulatoCore
import SimulatoMultislice
import SimulatoKitTestSupport

@Suite("Batch refusals and the Metal bounds' range")
struct BatchRefusalTests {
    @Test func aBatchOfTwoEnergiesIsRefused() throws {
        let lazy = try Potential(structure: aluminiumStructure(cells: (2, 2, 1)), nx: 32, ny: 32, sliceThickness: 2.025)
        let at200 = try ElectronBeam(energy: 200e3), at300 = try ElectronBeam(energy: 300e3)
        let a = Waves.planeWave(beam: at200, grid: lazy.grid), b = Waves.planeWave(beam: at300, grid: lazy.grid)
        let built = try lazy.build()
        for backend in [Backend.cpu, .metal] where backend == .cpu || Multislice.metalIsAvailable {
            #expect(throws: SimulatoError.mixedEnergies(first: 200e3, other: 300e3)) {
                try Multislice.runBatch([a, a, b], through: lazy, backend: backend)
            }
            #expect(throws: SimulatoError.mixedEnergies(first: 300e3, other: 200e3)) {
                try Multislice.runBatch([b, a], through: built, backend: backend)
            }
        }
        // One energy per batch: each wave is its single run (the energy taken from the wave, not the first one).
        let alone = try Multislice.run(b, through: built)
        #expect(try Multislice.runBatch([b, b], through: built).map(\.field) == [alone[0].field, alone[0].field])
    }

    @Test func theMetalBoundsSayWhatTheyCover() {
        let a = MetalAccuracy.registered
        #expect(a.covers(nx: 1024, ny: 1024, slices: 100) && a.covers(nx: 64, ny: 1024, slices: 1))
        #expect(!a.covers(nx: 2048, ny: 512, slices: 50) && !a.covers(nx: 512, ny: 1025, slices: 50))
        #expect(!a.covers(nx: 512, ny: 512, slices: 101))
        #expect(MetalAccuracy(sliceLimit: 500, gridLimit: 2048).covers(nx: 2048, ny: 2048, slices: 500))
    }
}

/// #59: the Metal path checks its sizes before making a buffer (a 32-bit scatter index trapped above 2^31; a buffer size
/// was never compared with the device's largest).
@Suite("Metal limits")
struct MetalLimitsTests {
    @Test func tensorBytesReportOverflow() {
        #expect(MetalLimits.tensorBytes([2, 3, 4], elementBytes: 8) == 192)
        #expect(MetalLimits.tensorBytes([], elementBytes: 4) == 4)
        #expect(MetalLimits.tensorBytes([1 << 32, 1 << 32], elementBytes: 8) == nil)
        #expect(MetalLimits.tensorBytes([1 << 31, 1 << 31], elementBytes: 8) == nil)
        #expect(MetalLimits.tensorBytes([-1, 4], elementBytes: 4) == nil)
    }

    @Test func scatterIndexFitsAnInt32() {
        #expect(MetalLimits.scatterIndexFits(elements: 3, points: 1024 * 1024))
        #expect(MetalLimits.scatterIndexFits(elements: 1, points: Int(Int32.max)))
        #expect(!MetalLimits.scatterIndexFits(elements: 2, points: Int(Int32.max)))
        #expect(!MetalLimits.scatterIndexFits(elements: 3000, points: 1 << 20))
    }
}


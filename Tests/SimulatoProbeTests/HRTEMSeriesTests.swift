//
//  HRTEMSeriesTests.swift
//  Role: the HRTEM image series (#88) without fixtures: every image equals `Waves.applying(_:)` then `intensity()` at
//        its defocus bit for bit, tiling repeats the image, and a non-finite defocus or a tile below 1 is refused.
//        On `.metal` every image is within `MetalAccuracy.registered.intensity` of `.cpu` (#89), over several batches.
//        The abTEM comparison is in SimulatoKitParityTests/HRTEMSeriesTests.
//

import Testing
import SimulatoCore
import SimulatoMultislice
import SimulatoProbe

@Suite("HRTEM image series")
struct HRTEMSeriesUnitTests {
    /// Two waves of different structure on one grid (a probe at two positions stands in for exit waves).
    static func waves() throws -> [Waves] {
        let beam = try ElectronBeam(energy: 200e3)
        let grid = try Grid(nx: 32, ny: 40, extentX: 8, extentY: 10)
        let probe = try Probe(beam: beam, grid: grid, aperture: Aperture(semiangleCutoff: 25))
        return [try probe.build(at: SIMD2(2, 3)), try probe.build(at: SIMD2(5.5, 7))]
    }

    @Test func everyImageEqualsApplyingTheFunctionAtItsDefocus() throws {
        let waves = try Self.waves()
        var lens = Aberrations()
        lens.C30 = 1.2e6
        lens.C12 = 20
        let ctf = try ContrastTransferFunction(aberrations: lens, semiangleCutoff: 30, focalSpread: 20, angularSpread: 0.3)
        let defocus = [-150.0, 0, 80, 300]
        let series = try ctf.images(of: waves, defocus: defocus)
        #expect(series.count == 2 && series.allSatisfy { $0.count == 4 })
        for (w, wave) in waves.enumerated() {
            for (d, df) in defocus.enumerated() {
                var one = ctf
                one.aberrations.defocus = df
                let alone = try wave.applying(one).intensity()
                #expect(series[w][d] == alone, "wave \(w) defocus \(df)")
            }
        }
        let tiled = try ctf.images(of: waves, defocus: [80], tile: (x: 2, y: 3))
        let repeated = try series[1][2].tiled(repetitionsX: 2, repetitionsY: 3)
        #expect(tiled[1][0] == repeated)
        #expect(try ctf.images(of: [], defocus: defocus).isEmpty)
        #expect(try ctf.images(of: waves, defocus: []) == [[], []])
    }

    @Test func badDefocusOrTileIsRefused() throws {
        let waves = try Self.waves()
        let ctf = try ContrastTransferFunction()
        #expect(throws: SimulatoError.self) { try ctf.images(of: waves, defocus: [0, .nan]) }
        #expect(throws: SimulatoError.self) { try ctf.images(of: waves, defocus: [.infinity]) }
        #expect(throws: ImageOperationError.invalidRepetitions(0, 1)) { try ctf.images(of: waves, defocus: [0], tile: (x: 0, y: 1)) }
    }

    @Test(.enabled(if: Multislice.metalIsAvailable, "no Metal device"))
    func onMetalEveryImageIsWithinTheRegisteredBoundOfTheCPU() throws {
        let waves = try Self.waves()
        var lens = Aberrations()
        lens.C30 = 1.2e6
        let ctf = try ContrastTransferFunction(aberrations: lens, semiangleCutoff: 30, focalSpread: 20)
        let defocus = [-150.0, -40, 0, 80, 300]
        let cpu = try ctf.images(of: waves, defocus: defocus)
        // Batches of 2 kernels: three batches per wave (2, 2, 1); `.standard` pads its one batch of 5 to 8.
        let small = ExecutionOptions(metalBatchBytes: 24 * 32 * 40 * 2)
        for options in [ExecutionOptions.standard, small] {
            let metal = try ctf.images(of: waves, defocus: defocus, backend: .metal, options: options)
            #expect(metal.count == 2 && metal.allSatisfy { $0.count == defocus.count })
            for w in waves.indices {
                for d in defocus.indices {
                    let a = cpu[w][d].values, b = metal[w][d].values
                    let peak = a.map(abs).max() ?? 0
                    let worst = zip(a, b).map { abs($0 - $1) }.max() ?? 0
                    #expect(peak > 0 && worst <= MetalAccuracy.registered.intensity * peak,
                            "wave \(w) defocus \(defocus[d]): \(worst) of \(peak)")
                    #expect(metal[w][d].nx == cpu[w][d].nx && metal[w][d].samplingX == cpu[w][d].samplingX)
                }
            }
        }
        let tiled = try ctf.images(of: waves, defocus: [80], tile: (x: 2, y: 3), backend: .metal)
        let one = try ctf.images(of: waves, defocus: [80], backend: .metal)
        let repeated = try one[1][0].tiled(repetitionsX: 2, repetitionsY: 3)
        #expect(tiled[1][0] == repeated)
        #expect(try ctf.images(of: waves, defocus: [], backend: .metal) == [[], []])
    }
}

//
//  HRTEMSeriesTests.swift
//  Role: the HRTEM image series (#88) without fixtures: every image equals `Waves.applying(_:)` then `intensity()` at
//        its defocus bit for bit, tiling repeats the image, and a non-finite defocus or a tile below 1 is refused.
//        The abTEM comparison is in SimulatoKitParityTests/HRTEMSeriesTests.
//

import Testing
import SimulatoCore
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
}

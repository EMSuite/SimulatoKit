//
//  HardeningTests.swift
//  Role: a line scan with an explicit huge `gpts` is refused, as the sampling path refuses it; a wave keeps its tilt through
//        a CTF.
//

import Testing
import SimulatoCore
import SimulatoProbe

@Suite("Probe hardening")
struct ProbeHardeningTests {
    @Test func hugeGptsIsRefused() throws {
        #expect(throws: SimulatoError.self) { try LineScan(end: SIMD2(1, 0), gpts: 2_000_000_000) }
        #expect(throws: SimulatoError.self) { try LineScan(end: SIMD2(1, 0), gpts: .max) }
        #expect(try LineScan(end: SIMD2(1, 0), gpts: 5).gpts == 5)
        // a line of zero length has NaN direction and angle
        let z = try LineScan(start: SIMD2(1, 1), end: SIMD2(1, 1), gpts: 3)
        #expect(z.direction.x.isNaN && z.angle.isNaN)
    }
}

@Suite("Transfer function keeps the wave's tilt")
struct CTFTiltTests {
    /// abTEM's `apply_ctf` keeps `base_tilt`; the kit dropped it (issue #60).
    @Test func aTiltedWaveKeepsItsTiltThroughACTF() throws {
        let grid = try Grid(nx: 32, ny: 32, extentX: 8, extentY: 8)
        let beam = try ElectronBeam(energy: 200e3)
        let tilted = Waves.planeWave(beam: beam, grid: grid, tilt: SIMD2(3, -2))
        var ab = Aberrations()
        ab.defocus = 50
        let out = try tilted.applying(ContrastTransferFunction(aberrations: ab))
        #expect(out.tilt == SIMD2(3, -2) && out.beam == beam && out.normalization == tilted.normalization)
    }
}

//
//  HRTEMSeriesBenchmarkTests.swift
//  Role: the benchmark row of the HRTEM series (#89), reported, not asserted: twenty defocus values (−100 to 90 Å) of
//        one exit wave of Al [001] 2 × 3 cells on 100 × 150 points (10 slices), 20 mrad CTF, `.cpu` and `.metal`,
//        median of 7 after a warm-up. Enabled only by SIMULATO_BENCHMARK=1 (or `metal`).
//

import Foundation
import Testing
import SimulatoCore
import SimulatoMultislice
import SimulatoProbe

@Suite("HRTEM series benchmark (SIMULATO_BENCHMARK=1)",
       .enabled(if: ["1", "metal"].contains(ProcessInfo.processInfo.environment["SIMULATO_BENCHMARK"] ?? ""),
                "set SIMULATO_BENCHMARK=1 to run"),
       .enabled(if: Multislice.metalIsAvailable, "no Metal device"))
struct HRTEMSeriesBenchmarkTests {
    @Test func twentyDefocusValuesOfOneUnit() throws {
        let cell = try AtomicStructure(box: SIMD3(4.05, 4.05, 4.05), atoms: [
            .init(atomicNumber: 13, position: SIMD3(0, 0, 0)), .init(atomicNumber: 13, position: SIMD3(0, 2.025, 2.025)),
            .init(atomicNumber: 13, position: SIMD3(2.025, 0, 2.025)), .init(atomicNumber: 13, position: SIMD3(2.025, 2.025, 0)),
        ])
        let potential = try Potential(structure: cell.repeated(2, 3, 5), nx: 100, ny: 150, sliceThickness: 2.025)
        let beam = try ElectronBeam(energy: 200e3)
        let exit = try Multislice.run(.planeWave(beam: beam, grid: potential.grid), through: potential)
        let ctf = try ContrastTransferFunction(semiangleCutoff: 20)
        let defocus = (0..<20).map { -100 + 10 * Double($0) }
        let clock = ContinuousClock()
        var row: [String] = []
        for backend in [Backend.cpu, .metal] {
            _ = try ctf.images(of: exit, defocus: defocus, backend: backend)
            var times: [Double] = []
            for _ in 0..<7 {
                let t = try clock.measure { _ = try ctf.images(of: exit, defocus: defocus, backend: backend) }
                times.append(Double(t.components.seconds) * 1e3 + Double(t.components.attoseconds) * 1e-15)
            }
            times.sort()
            row.append(String(format: "%@ %.2f ms (%.2f-%.2f)", "\(backend)", times[3], times[0], times[6]))
        }
        print("BENCH HRTEM series, 20 defocus, 100x150: " + row.joined(separator: "; "))
    }
}

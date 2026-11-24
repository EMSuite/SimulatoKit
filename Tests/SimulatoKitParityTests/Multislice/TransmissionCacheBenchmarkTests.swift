//
//  TransmissionCacheBenchmarkTests.swift
//  Role: the benchmark row of the transmission cache (#122), reported, not asserted: a static Al [001] slab of 250
//        slices on 1024² (5 slices per period, 50 periods), one plane wave on `.metal` through the unbuilt `Potential`
//        (preparation included), cached and with the cache off, the same through the built slices, and a slab where
//        no slice repeats; median of 7 after a warm-up, and the largest difference between cached and uncached. The
//        issue's bar: cached under 0.30 s. Enabled only by SIMULATO_BENCHMARK=1.
//

import Foundation
import Testing
import SimulatoCore
import SimulatoMultislice
import SimulatoKitTestSupport

@Suite("Transmission cache benchmark (SIMULATO_BENCHMARK=1)",
       .enabled(if: ["1", "metal"].contains(ProcessInfo.processInfo.environment["SIMULATO_BENCHMARK"] ?? ""),
                "set SIMULATO_BENCHMARK=1 to run"),
       .enabled(if: Multislice.metalIsAvailable, "no Metal device"))
struct TransmissionCacheBenchmarkTests {
    @Test func aStatic250SliceRunAt1024() throws {
        let sites: [CrystalCell.Site] = [SIMD3<Double>(0, 0, 0), SIMD3(0, 0.5, 0.5), SIMD3(0.5, 0, 0.5), SIMD3(0.5, 0.5, 0)]
            .map { CrystalCell.Site(atomicNumber: 13, fractional: $0) }
        let al = try CrystalCell(a: 4.05, b: 4.05, c: 4.05, alpha: 90, beta: 90, gamma: 90, sites: sites)
        let slab = try SlabBuilder.slab(of: al, zoneAxis: SIMD3(0, 0, 1), lateralSize: SIMD2(51, 51),
                                        thickness: 4.05 * 50, sliceThickness: 1)
        let potential = try Potential(structure: slab.structure, nx: 1024, ny: 1024, sliceThicknesses: slab.sliceThicknesses)
        let beam = try ElectronBeam(energy: 200e3)
        let wave = Waves.planeWave(beam: beam, grid: potential.grid)
        // The same atoms nudged by 1e−6 Å per atom along x: no slice repeats (a frozen-phonon configuration's case).
        let nudged = try AtomicStructure(box: slab.structure.box, atoms: slab.structure.atoms.enumerated().map {
            (q: Int, atom: AtomicStructure.Atom) -> AtomicStructure.Atom in
            var p = atom.position
            p.x += 1e-6 * Double(q % 1000)
            return AtomicStructure.Atom(atomicNumber: atom.atomicNumber, position: p)
        })
        let distinctAll = try Potential(structure: nudged, nx: 1024, ny: 1024, sliceThicknesses: slab.sliceThicknesses)
        let built = try potential.build()
        let clock = ContinuousClock()
        var row: [String] = ["\(potential.sliceCount) slices, \(Set(potential.firstEqualSlices()).count) distinct"]
        var exits: [ComplexField] = []
        let cached = ExecutionOptions.standard
        let uncached = ExecutionOptions(metalTransmissionCacheBytes: 0)
        let cases: [(label: String, run: () throws -> ComplexField)] = [
            ("cached", { try Multislice.run(wave, through: potential, backend: .metal, options: cached)[0].field }),
            ("uncached", { try Multislice.run(wave, through: potential, backend: .metal, options: uncached)[0].field }),
            ("built slices cached", { try Multislice.run(wave, through: built, backend: .metal, options: cached)[0].field }),
            ("built slices uncached", { try Multislice.run(wave, through: built, backend: .metal, options: uncached)[0].field }),
            ("no slice repeats", { try Multislice.run(wave, through: distinctAll, backend: .metal)[0].field }),
        ]
        for (label, run) in cases {
            exits.append(try run())
            var times: [Double] = []
            for _ in 0..<7 {
                let t = try clock.measure { _ = try run() }
                times.append(Double(t.components.seconds) + Double(t.components.attoseconds) * 1e-18)
            }
            times.sort()
            row.append(String(format: "%@ %.3f s (%.3f-%.3f)", label, times[3], times[0], times[6]))
        }
        row.append(String(format: "cached vs uncached: max diff / max %.1e", relMax(exits[0], exits[1])))
        print("BENCH transmission cache, Al [001] 1024², .metal: " + row.joined(separator: "; "))
    }
}

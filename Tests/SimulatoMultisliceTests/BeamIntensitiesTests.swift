//
//  BeamIntensitiesTests.swift
//  Role: `BeamIntensities` by truths that need no fixture: the integer map between the crystal's hkl and the slab's grid,
//        the incident beam, the forbidden 002 beam of Si [110] (zero kinematically, growing from zero dynamically), and
//        the refusals. The pins against abTEM are in SimulatoKitParityTests.
//

import Foundation
import Testing
import SimulatoCore
import SimulatoMultislice

@Suite("Beam intensities by hkl")
struct BeamIntensitiesTests {
    /// Si, diamond, a = 5.431 Å, the eight sites of the cubic cell.
    static func silicon() throws -> CrystalCell {
        let f: [SIMD3<Double>] = [SIMD3(0, 0, 0), SIMD3(0.25, 0.25, 0.25), SIMD3(0, 0.5, 0.5), SIMD3(0.25, 0.75, 0.75),
                                  SIMD3(0.5, 0, 0.5), SIMD3(0.75, 0.25, 0.75), SIMD3(0.5, 0.5, 0), SIMD3(0.75, 0.75, 0.25)]
        return try CrystalCell(a: SIMD3(5.431, 0, 0), b: SIMD3(0, 5.431, 0), c: SIMD3(0, 0, 5.431),
                               sites: f.map { CrystalCell.Site(atomicNumber: 14, fractional: $0) })
    }

    /// Si [110], `periods` deep, its exit waves after the incident plane and after each of `stored` periods.
    static func silicon110(periods: Int, stored: [Int], gpts: (Int, Int) = (28, 40)) throws -> (Slab, BeamIntensities) {
        let si = try silicon()
        let unit = try SlabBuilder.orthogonalUnit(of: si, zoneAxis: SIMD3(1, 1, 0))
        let slab = try SlabBuilder.slab(of: si, zoneAxis: SIMD3(1, 1, 0), thickness: unit.structure.box.z * Double(periods))
        let k = slab.sliceThicknesses.count / slab.repeats.z
        let planes = ExitPlanes.afterSlices([-1] + stored.map { $0 * k - 1 })
        let potential = try Potential(structure: slab.structure, nx: gpts.0, ny: gpts.1,
                                      sliceThicknesses: slab.sliceThicknesses)
        let waves = try Multislice.run(Waves.planeWave(beam: try ElectronBeam(energy: 200e3), grid: potential.grid),
                                       through: potential, exitPlanes: planes)
        return (slab, try BeamIntensities(exitWaves: waves, of: slab, exitPlanes: planes))
    }

    @Test func crystalHKLMapsToTheSlabsGridExactly() throws {
        let si = try Self.silicon()
        for zone in [SIMD3(1, 1, 0), SIMD3(1, 1, 1), SIMD3(0, 0, 1)] {
            let slab = try SlabBuilder.slab(of: si, zoneAxis: zone, lateralSize: SIMD2(20, 20), thickness: 10)
            #expect(slab.repeats.x > 1 && slab.repeats.y > 1, "\(zone) repeats \(slab.repeats)")
            for h in -4...4 {
                for k in -4...4 {
                    for l in -4...4 {
                        let hkl = SIMD3(h, k, l)
                        let inZone = h * zone.x + k * zone.y + l * zone.z == 0
                        let found = BeamIntensities.frequencyIndex(of: hkl, in: slab)
                        #expect((found != nil) == inZone, "\(zone) \(hkl)")
                        guard let index = found else { continue }
                        #expect(index.x % slab.repeats.x == 0 && index.y % slab.repeats.y == 0, "\(zone) \(hkl) at \(index)")
                        #expect(BeamIntensities.hkl(atFrequencyIndex: index, in: slab) == hkl, "\(zone) \(hkl) back")
                    }
                }
            }
            // A frequency between the crystal's reflections holds none.
            #expect(BeamIntensities.hkl(atFrequencyIndex: SIMD2(1, 0), in: slab) == nil, "\(zone) (1, 0)")
        }
    }

    @Test func kinematic002OfSilicon110IsZeroAnd111IsNot() throws {
        let si = try Self.silicon()
        let unit = try SlabBuilder.orthogonalUnit(of: si, zoneAxis: SIMD3(1, 1, 0))
        let box = unit.structure.box
        func structureFactor(_ hkl: SIMD3<Int>) -> Double {
            let v = unit.supercell
            let mx = Double(v[0].x * hkl.x + v[0].y * hkl.y + v[0].z * hkl.z)
            let my = Double(v[1].x * hkl.x + v[1].y * hkl.y + v[1].z * hkl.z)
            let mz = Double(v[2].x * hkl.x + v[2].y * hkl.y + v[2].z * hkl.z)
            var re = 0.0, im = 0.0
            for atom in unit.structure.atoms {
                let p = atom.position
                let phase = 2 * Double.pi * (mx * p.x / box.x + my * p.y / box.y + mz * p.z / box.z)
                re += cos(phase)
                im += sin(phase)
            }
            return (re * re + im * im).squareRoot()
        }
        #expect(structureFactor(SIMD3(0, 0, 2)) <= 1e-9, "F(002) = \(structureFactor(SIMD3(0, 0, 2)))")
        #expect(structureFactor(SIMD3(1, -1, 1)) > 1, "F(1-11) = \(structureFactor(SIMD3(1, -1, 1)))")
        #expect(structureFactor(SIMD3(2, -2, 0)) > 1, "F(2-20) = \(structureFactor(SIMD3(2, -2, 0)))")
    }

    @Test func forbidden002OfSilicon110GrowsFromZero() throws {
        let stored = [1, 2, 3, 4, 6, 8]
        let (slab, result) = try Self.silicon110(periods: 8, stored: stored)
        #expect(result.thicknesses.count == stored.count + 1)
        #expect(result.thicknesses[0] == 0)
        for (t, p) in zip(result.thicknesses.dropFirst(), stored) {
            #expect(abs(t - Double(p) * slab.period) <= 1e-9, "thickness \(t)")
        }
        let i002 = try #require(result.intensities(of: SIMD3(0, 0, 2)))
        let i000 = try #require(result.intensities(of: SIMD3(0, 0, 0)))
        #expect(abs(i000[0] - 1) <= 1e-12, "the incident beam \(i000[0])")
        #expect(i002[0] <= 1e-20, "002 in the incident wave \(i002[0])")
        for t in 1..<i002.count { #expect(i002[t] > i002[t - 1], "002 grows: \(i002)") }
        #expect(i002[i002.count - 1] > 1e-4, "002 after 8 periods \(i002[i002.count - 1])")
        // Beams come by |g|: the direct beam first.
        #expect(result.beams.first?.hkl == SIMD3(0, 0, 0))
        #expect(result.beams.allSatisfy { $0.hkl.x + $0.hkl.y == 0 }, "only the zero-order Laue zone of [110]")
    }

    @Test func refusesWavesThatDoNotFitTheSlab() throws {
        let si = try Self.silicon()
        let slab = try SlabBuilder.slab(of: si, zoneAxis: SIMD3(1, 1, 0), thickness: 8)
        let beam = try ElectronBeam(energy: 200e3)
        let grid = try Grid(nx: 28, ny: 40, extentX: slab.structure.box.x, extentY: slab.structure.box.y)
        let wave = Waves.planeWave(beam: beam, grid: grid)
        #expect(throws: SimulatoError.self) {
            _ = try BeamIntensities(exitWaves: [wave, wave], of: slab, exitPlanes: .last)
        }
        #expect(throws: SimulatoError.self) { _ = try BeamIntensities(exitWaves: [], of: slab, exitPlanes: .last) }
        let other = try Grid(nx: 28, ny: 40, extentX: slab.structure.box.x * 2, extentY: slab.structure.box.y)
        #expect(throws: SimulatoError.gridMismatch) {
            _ = try BeamIntensities(exitWaves: [Waves.planeWave(beam: beam, grid: other)], of: slab, exitPlanes: .last)
        }
        #expect(throws: SimulatoError.self) {
            _ = try BeamIntensities(exitWaves: [wave], of: slab, exitPlanes: .afterSlices([99]))
        }
    }
}

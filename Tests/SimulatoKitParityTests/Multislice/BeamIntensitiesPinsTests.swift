//
//  BeamIntensitiesPinsTests.swift
//  Role: `BeamIntensities` against abTEM 1.0.10 `index_diffraction_spots` with the lattice's `centering`
//        (tools/pins/beam-intensities/pins.py): Si [110], Si [111], Mg [0001] and Si [001] at 200 kV, five thicknesses from
//        about 2 to 100 nm; no beam the F lattice forbids (SimulatoKit#145). The hkl of every beam and its pixel
//        exact; every intensity within 1e-10 of the incident beam (the kit's exit-wave pin, 1e-10 of max with abTEM's
//        float32 pi). The thin Si [110] series holds the forbidden 002 beam growing from zero.
//

import Foundation
import Testing
import SimulatoCore
import SimulatoMultislice

enum BeamIntensityPins {
    static let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Fixtures")

    struct Loaded: @unchecked Sendable {  // immutable after load
        let json: [String: Any]
    }

    static let loaded: Loaded = {
        let json = try! JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent("beam-intensities-pins.json"))) as! [String: Any]
        return Loaded(json: json)
    }()

    static var json: [String: Any] { loaded.json }
    static var cases: [[String: Any]] { json["cases"] as! [[String: Any]] }
    static var truth: [String: Any] { json["truth"] as! [String: Any] }

    static func cell(_ name: String) throws -> CrystalCell {
        let s = (json["structures"] as! [String: Any])[name] as! [String: Any]
        let rows = (s["cell"] as! [[Double]]).map { SIMD3($0[0], $0[1], $0[2]) }
        let sites = (s["sites"] as! [[Double]]).map {
            CrystalCell.Site(atomicNumber: Int($0[0]), fractional: SIMD3($0[1], $0[2], $0[3]))
        }
        return try CrystalCell(a: rows[0], b: rows[1], c: rows[2], sites: sites)
    }

    static func ints(_ a: Any) -> SIMD3<Int> {
        let v = a as! [Int]
        return SIMD3(v[0], v[1], v[2])
    }

    /// The slab of `structure` along `zone`, `periods` deep, its exit waves after the given periods, and their beams.
    static func run(structure: String, zone: SIMD3<Int>, periods: [Int], gpts: [Int]) throws -> (Slab, BeamIntensities) {
        let crystal = try Self.cell(structure)
        let unit = try SlabBuilder.orthogonalUnit(of: crystal, zoneAxis: zone)
        let deepest = periods.max() ?? 1
        let slab = try SlabBuilder.slab(of: crystal, zoneAxis: zone, thickness: unit.structure.box.z * Double(deepest),
                                        sliceThickness: json["slice_thickness"] as! Double)
        let k = slab.sliceThicknesses.count / slab.repeats.z
        let planes = ExitPlanes.afterSlices(periods.map { $0 * k - 1 })
        let potential = try Potential(structure: slab.structure, nx: gpts[0], ny: gpts[1],
                                      sliceThicknesses: slab.sliceThicknesses,
                                      parameterization: .lobato(LobatoParameterization(piConstant: .abTEMFloat32)))
        let beam = try ElectronBeam(energy: json["energy"] as! Double)
        let waves = try Multislice.run(Waves.planeWave(beam: beam, grid: potential.grid), through: potential,
                                       exitPlanes: planes)
        return (slab, try BeamIntensities(exitWaves: waves, of: slab, exitPlanes: planes))
    }
}

@Suite("Beam intensities by hkl against abTEM")
struct BeamIntensitiesPinsTests {
    struct Case: Sendable, CustomTestStringConvertible {
        static let all = BeamIntensityPins.cases.indices.map(Case.init)
        let index: Int
        var testDescription: String {
            let c = BeamIntensityPins.cases[index]
            return "\(c["structure"] as! String) \(c["zone"] as! [Int])"
        }
    }

    @Test(arguments: Case.all)
    func beamsEqualAbTEMsIndexedSpots(c: Case) throws {
        let pin = BeamIntensityPins.cases[c.index]
        let tag = c.testDescription
        let (slab, result) = try BeamIntensityPins.run(structure: pin["structure"] as! String,
                                                       zone: BeamIntensityPins.ints(pin["zone"]!),
                                                       periods: pin["periods"] as! [Int], gpts: pin["gpts"] as! [Int])
        let supercell = (pin["supercell"] as! [[Int]]).map { SIMD3($0[0], $0[1], $0[2]) }
        #expect(slab.unit.supercell == supercell, "\(tag) supercell \(slab.unit.supercell)")
        #expect(slab.sliceThicknesses.count / slab.repeats.z == pin["slices_per_period"] as! Int, "\(tag) slices per period")

        let thicknesses = pin["thicknesses"] as! [Double]
        #expect(result.thicknesses.count == thicknesses.count)
        for (got, want) in zip(result.thicknesses, thicknesses) {
            #expect(abs(got - want) <= 1e-12 * want, "\(tag) thickness \(got) against \(want) Å")
        }

        // The hkl and the pixel of every beam exact, as the same set.
        let hkl = (pin["hkl"] as! [[Int]]).map { SIMD3($0[0], $0[1], $0[2]) }
        let pixels = (pin["pixels"] as! [[Int]]).map { SIMD2($0[0], $0[1]) }
        #expect(result.beams.count == hkl.count, "\(tag) \(result.beams.count) beams against \(hkl.count)")
        #expect(Set(result.beams.map(\.hkl)) == Set(hkl), "\(tag) the hkl sets differ")
        for (h, p) in zip(hkl, pixels) {
            #expect(BeamIntensities.frequencyIndex(of: h, in: slab) == p, "\(tag) \(h) at \(p)")
            #expect(BeamIntensities.hkl(atFrequencyIndex: p, in: slab) == h, "\(tag) \(p) is \(h)")
        }

        // Every intensity within 1e-10 of the incident beam.
        let want = pin["intensities"] as! [[Double]]
        var worst = 0.0
        for (j, h) in hkl.enumerated() {
            guard let got = result.intensities(of: h) else { continue }
            for t in want.indices { worst = max(worst, abs(got[t] - want[t][j])) }
        }
        #expect(worst <= 1e-10, "\(tag) worst intensity difference \(worst)")
    }

    /// Si [110]: 002 is zero kinematically (the diamond glide); dynamically it grows from zero with thickness.
    @Test func siliconForbidden002GrowsFromZeroAsAbTEMs() throws {
        let pin = BeamIntensityPins.truth
        let periods = pin["periods"] as! [Int]
        let pinned = BeamIntensityPins.cases.first(where: { ($0["structure"] as! String) == "Si" && ($0["zone"] as! [Int]) == [1, 1, 0] })!
        let (_, result) = try BeamIntensityPins.run(structure: "Si", zone: SIMD3(1, 1, 0), periods: periods,
                                                    gpts: pinned["gpts"] as! [Int])
        let got = try #require(result.intensities(of: SIMD3(0, 0, 2)))
        let want = pin["si110_002"] as! [Double]
        #expect(got.count == want.count)
        for (g, w) in zip(got, want) { #expect(abs(g - w) <= 1e-10, "002: \(g) against abTEM's \(w)") }
        for t in 1..<got.count { #expect(got[t] > got[t - 1], "002 grows: \(got)") }
        #expect(got[0] > 0)
    }
}

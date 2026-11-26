//
//  BeamMatrix.swift
//  Role: the beam matrix (tools/pins/beam-matrix/pins.py): eleven crystals, their zone axes and 80, 200 and 300 kV
//        against abTEM 1.0.10's `index_diffraction_spots` with the lattice's centring. One runner for both sides: the
//        fast unit tests take one case per crystal at 200 kV, the parity target every case.
//

import Foundation
import SimulatoCore
import SimulatoMultislice

package enum BeamMatrix {
    struct File: Decodable {
        struct Structure: Decodable {
            let cell: [[Double]]
            let centering: String
            let sites: [[Double]]
        }
        struct Entry: Decodable {
            let structure: String
            let zone: [Int]
            let energy: Double
            let fast: Bool
            let supercell: [[Int]]
            let periods: [Int]
            let slicesPerPeriod: Int
            let gpts: [Int]
            let thicknesses: [Double]
            let hkl: [[Int]]
            let intensities: [[Double]]

            enum CodingKeys: String, CodingKey {
                case structure, zone, energy, fast, supercell, periods, gpts, thicknesses, hkl, intensities
                case slicesPerPeriod = "slices_per_period"
            }
        }
        let sliceThickness: Double
        let structures: [String: Structure]
        let cases: [Entry]

        enum CodingKeys: String, CodingKey {
            case structures, cases
            case sliceThickness = "slice_thickness"
        }
    }

    static let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("SimulatoKitParityTests").appendingPathComponent("Fixtures")
        .appendingPathComponent("beam-matrix-pins.json")

    static let file: File = {
        // A missing or unreadable fixture is a broken checkout: stop with its path.
        guard let data = try? Data(contentsOf: url), let file = try? JSONDecoder().decode(File.self, from: data) else {
            fatalError("cannot read \(url.path)")
        }
        return file
    }()

    /// The indices of every case.
    package static var all: [Int] { Array(file.cases.indices) }
    /// The indices of the fast cases: one per crystal at 200 kV.
    package static var fast: [Int] { file.cases.indices.filter { file.cases[$0].fast } }

    /// "Si [0 0 1] 200 kV".
    package static func name(_ index: Int) -> String {
        let c = file.cases[index]
        return "\(c.structure) [\(c.zone.map(String.init).joined(separator: " "))] \(Int(c.energy / 1e3)) kV"
    }

    /// Whether the centring letter's reflection condition allows hkl (P, F, I, C, A, B), written out apart from the kit.
    static func reflects(_ h: SIMD3<Int>, _ centering: String) -> Bool {
        func even(_ n: Int) -> Bool { n % 2 == 0 }
        switch centering {
        case "F": return even(h.x - h.y) && even(h.y - h.z)
        case "I": return even(h.x + h.y + h.z)
        case "C": return even(h.x + h.y)
        case "A": return even(h.y + h.z)
        case "B": return even(h.x + h.z)
        default: return true
        }
    }

    /// Runs case `index` as the generator built it (the kit's slab, Lobato with abTEM's float32 π, the stored exit
    /// planes) and returns what differs: the unit, the slices, the thicknesses, a beam the lattice forbids, beams that
    /// sum past the incident beam, the hkl set against abTEM's, and any intensity more than 1e−10 of the incident beam
    /// from abTEM's (the kit's exit-wave pin). Empty when the case agrees.
    package static func check(_ index: Int) throws -> [String] {
        let pin = file.cases[index]
        let tag = name(index)
        let s = try requireStructure(pin.structure)
        let rows = s.cell.map { SIMD3($0[0], $0[1], $0[2]) }
        let sites = s.sites.map { CrystalCell.Site(atomicNumber: Int($0[0]), fractional: SIMD3($0[1], $0[2], $0[3])) }
        let crystal = try CrystalCell(a: rows[0], b: rows[1], c: rows[2], sites: sites)
        let zone = SIMD3(pin.zone[0], pin.zone[1], pin.zone[2])
        let unit = try SlabBuilder.orthogonalUnit(of: crystal, zoneAxis: zone)
        let deepest = pin.periods.max() ?? 1
        let slab = try SlabBuilder.slab(of: crystal, zoneAxis: zone, thickness: unit.structure.box.z * Double(deepest),
                                        sliceThickness: file.sliceThickness)
        var failures: [String] = []
        let supercell = pin.supercell.map { SIMD3($0[0], $0[1], $0[2]) }
        guard slab.unit.supercell == supercell else {
            return ["\(tag): supercell \(slab.unit.supercell) against the generator's \(supercell)"]
        }
        let k = slab.sliceThicknesses.count / slab.repeats.z
        if k != pin.slicesPerPeriod { failures.append("\(tag): \(k) slices per period against \(pin.slicesPerPeriod)") }
        let planes = ExitPlanes.afterSlices(pin.periods.map { $0 * k - 1 })
        let potential = try Potential(structure: slab.structure, nx: pin.gpts[0], ny: pin.gpts[1],
                                      sliceThicknesses: slab.sliceThicknesses,
                                      parameterization: .lobato(LobatoParameterization(piConstant: .abTEMFloat32)))
        let waves = try Multislice.run(Waves.planeWave(beam: try ElectronBeam(energy: pin.energy), grid: potential.grid),
                                       through: potential, exitPlanes: planes)
        let result = try BeamIntensities(exitWaves: waves, of: slab, exitPlanes: planes)

        for (got, want) in zip(result.thicknesses, pin.thicknesses) where abs(got - want) > 1e-9 * want {
            failures.append("\(tag): thickness \(got) against \(want) Å")
        }
        // The lattice's own rules, apart from abTEM.
        let forbidden = result.beams.map(\.hkl).filter { !reflects($0, s.centering) }
        if !forbidden.isEmpty {
            failures.append("\(tag): \(forbidden.count) beams the \(s.centering) lattice forbids, e.g. \(forbidden.prefix(3))")
        }
        for (t, row) in result.intensities.enumerated() where row.reduce(0, +) > 1 + 1e-6 {
            failures.append("\(tag): the beams at \(result.thicknesses[t]) Å sum to \(row.reduce(0, +))")
        }
        // abTEM's indexed beams: the same hkl set, every intensity within 1e-10 of the incident beam.
        let hkl = pin.hkl.map { SIMD3($0[0], $0[1], $0[2]) }
        let got = Set(result.beams.map(\.hkl)), want = Set(hkl)
        if got != want {
            let extra = got.subtracting(want), missing = want.subtracting(got)
            failures.append("\(tag): \(extra.count) beams abTEM does not list (e.g. \(extra.prefix(3))), "
                            + "\(missing.count) it lists that the kit does not (e.g. \(missing.prefix(3)))")
        }
        var worst = 0.0, at = SIMD3(0, 0, 0)
        for (j, h) in hkl.enumerated() {
            guard let mine = result.intensities(of: h) else { continue }
            for t in pin.intensities.indices where abs(mine[t] - pin.intensities[t][j]) > worst {
                worst = abs(mine[t] - pin.intensities[t][j])
                at = h
            }
        }
        if worst > 1e-10 { failures.append("\(tag): intensity \(at) differs from abTEM's by \(worst)") }
        return failures
    }

    static func requireStructure(_ name: String) throws -> File.Structure {
        guard let s = file.structures[name] else { throw SimulatoError.invalidShape("no structure \(name) in the matrix") }
        return s
    }
}

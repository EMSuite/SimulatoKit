//
//  OccupancyDrawTests.swift
//  Role: partly occupied sites drawn per configuration (#120). abTEM has no partial occupancy, so the reference is the
//        closed form: the mean projected potential over the draws equals the occupancy-weighted potential within three
//        standard errors. Also: one atom at most per shared site, each atom kept at its occupancy, reproducible draws
//        (masks pinned from an independent Python port of the generator), the refusals.
//

import Foundation
import Testing
import SimulatoCore
import SimulatoMultislice

@Suite("Partly occupied sites")
struct OccupancyDrawTests {
    /// O whole at (1, 1, 1); Si 0.6 alone at (3, 1, 1); Fe 0.3 and Ni 0.5 sharing (3, 4, 1); a 6 × 6 × 2 Å box.
    static func structure() throws -> AtomicStructure {
        try AtomicStructure(box: SIMD3(6, 6, 2), atoms: [
            .init(atomicNumber: 8, position: SIMD3(1, 1, 1)),
            .init(atomicNumber: 14, position: SIMD3(3, 1, 1), occupancy: 0.6),
            .init(atomicNumber: 26, position: SIMD3(3, 4, 1), occupancy: 0.3),
            .init(atomicNumber: 28, position: SIMD3(3, 4, 1), occupancy: 0.5),
        ])
    }

    @Test func sitesGroupTheAtomsThatShareAPosition() throws {
        let draw = try OccupancyDraw(structure: Self.structure())
        #expect(draw.sites == [[1], [2, 3]])
        #expect(!draw.isEmpty)
        let whole = try AtomicStructure(box: SIMD3(6, 6, 2), atoms: [.init(atomicNumber: 8, position: SIMD3(1, 1, 1))])
        #expect(try OccupancyDraw(structure: whole).isEmpty)
    }

    /// The kept atoms of seed 2026, configurations 0 ... 5, from `tools/pins/occupancy/draw.py` (the generator ported to
    /// Python): the same masks on every platform.
    @Test func drawsAreReproducibleFromSeedAndConfiguration() throws {
        let draw = try OccupancyDraw(structure: Self.structure())
        let expected: [[Bool]] = [
            [true, true, false, true], [true, false, false, false], [true, false, false, true],
            [true, true, false, false], [true, false, false, true], [true, true, false, true],
        ]
        for (c, mask) in expected.enumerated() {
            #expect(draw.kept(seed: 2026, configuration: c) == mask, "configuration \(c)")
            #expect(draw.kept(seed: 2026, configuration: c) == draw.kept(seed: 2026, configuration: c))
        }
    }

    /// Over 4000 configurations: never Fe and Ni together, each atom kept at its occupancy within four binomial standard
    /// errors, the whole atom always.
    @Test func eachSiteHoldsOneAtomAtMostAndEachAtomItsOccupancy() throws {
        let draw = try OccupancyDraw(structure: Self.structure())
        let n = 4000
        var counts = [0, 0, 0, 0]
        for c in 0..<n {
            let keep = draw.kept(seed: 7, configuration: c)
            #expect(!(keep[2] && keep[3]), "configuration \(c) put Fe and Ni on one site")
            for i in counts.indices where keep[i] { counts[i] += 1 }
        }
        #expect(counts[0] == n)
        for (i, p) in [(1, 0.6), (2, 0.3), (3, 0.5)] {
            let f = Double(counts[i]) / Double(n)
            let se = (p * (1 - p) / Double(n)).squareRoot()
            #expect(abs(f - p) <= 4 * se, "atom \(i): kept \(f), occupancy \(p)")
        }
    }

    /// The closed form: the projected potential is linear in the atoms, so its mean over 2000 drawn configurations equals
    /// Σ occupancy_i V_i (V_i the potential of atom i alone) within three standard errors at every pixel.
    @Test func meanPotentialIsTheOccupancyWeightedPotential() throws {
        let s = try Self.structure()
        var weighted = [Double](repeating: 0, count: 24 * 24)
        for var atom in s.atoms {
            let p = atom.occupancy
            atom.occupancy = 1
            let alone = try ProjectedPotential(structure: AtomicStructure(box: s.box, atoms: [atom]), nx: 24, ny: 24,
                                               sliceThickness: 2)
            for (k, x) in alone.slices[0].values.enumerated() { weighted[k] += p * x }
        }
        // Deviations from the closed form, accumulated so the whole atom's constant part cancels before squaring.
        let n = 2000
        let phonons = try FrozenPhonons(structure: s, siteSigmas: [0, 0, 0, 0], configurations: n, seed: 2026)
        var sum = [Double](repeating: 0, count: 24 * 24)
        var sumSquares = sum
        for c in 0..<n {
            let v = try ProjectedPotential(structure: phonons.configuration(c), nx: 24, ny: 24, sliceThickness: 2)
            for (k, x) in v.slices[0].values.enumerated() {
                let d = x - weighted[k]
                sum[k] += d
                sumSquares[k] += d * d
            }
        }
        let largest = weighted.map(abs).max() ?? 0
        for k in weighted.indices {
            let mean = sum[k] / Double(n)
            let variance = max(0, sumSquares[k] / Double(n) - mean * mean) * Double(n) / Double(n - 1)
            let se = (variance / Double(n)).squareRoot()
            #expect(abs(mean) <= 3 * se + 1e-12 * largest, "pixel \(k): mean deviation \(mean), standard error \(se)")
        }
    }

    /// Drawn configurations hold whole atoms; the displacements of the same seed do not change when atoms are partly
    /// occupied.
    @Test func configurationsHoldWholeAtomsAndKeepTheirDisplacements() throws {
        let s = try Self.structure()
        let phonons = try FrozenPhonons(structure: s, siteSigmas: [0.1, 0.1, 0.1, 0.1], configurations: 3, seed: 11)
        var whole = s.atoms
        for i in whole.indices { whole[i].occupancy = 1 }
        let still = try FrozenPhonons(structure: AtomicStructure(box: s.box, atoms: whole), siteSigmas: [0.1, 0.1, 0.1, 0.1],
                                      configurations: 3, seed: 11)
        #expect(phonons.displacements == still.displacements)
        #expect(still.occupancy.isEmpty)
        for c in 0..<3 {
            let atoms = try phonons.configuration(c).atoms
            let keep = phonons.occupancy.kept(seed: 11, configuration: c)
            #expect(atoms.count == keep.filter { $0 }.count)
            #expect(atoms.allSatisfy { $0.occupancy == 1 })
        }
    }

    @Test func refusals() throws {
        // A shared site filled beyond 1 names its first atom and its total.
        let over = try AtomicStructure(box: SIMD3(4, 4, 4), atoms: [
            .init(atomicNumber: 26, position: SIMD3(1, 1, 1), occupancy: 0.7),
            .init(atomicNumber: 28, position: SIMD3(1, 1, 1 + 5e-5), occupancy: 0.6),
        ])
        #expect(throws: SimulatoError.invalidOccupancy(atom: 0, occupancy: 0.7 + 0.6)) { try OccupancyDraw(structure: over) }
        // A whole atom on a partly occupied site fills it beyond 1 too.
        let crowded = try AtomicStructure(box: SIMD3(4, 4, 4), atoms: [
            .init(atomicNumber: 26, position: SIMD3(1, 1, 1)),
            .init(atomicNumber: 28, position: SIMD3(1, 1, 1), occupancy: 0.5),
        ])
        #expect(throws: SimulatoError.invalidOccupancy(atom: 0, occupancy: 1.5)) {
            try FrozenPhonons(structure: crowded, siteSigmas: [0, 0], configurations: 1, seed: 1)
        }
        // Occupancies outside (0, 1].
        for o in [0, -0.1, 1.5, Double.nan] {
            let error = #expect(throws: SimulatoError.self) {
                try AtomicStructure(box: SIMD3(4, 4, 4), atoms: [.init(atomicNumber: 8, position: .zero, occupancy: o)])
            }
            guard case .invalidOccupancy(0, _)? = error else {
                Issue.record("occupancy \(o): \(String(describing: error))")
                continue
            }
        }
        // A potential takes whole atoms.
        #expect(throws: SimulatoError.partlyOccupiedAtom(index: 1, occupancy: 0.6)) {
            try Potential(structure: Self.structure(), nx: 24, ny: 24, sliceThickness: 2)
        }
    }
}

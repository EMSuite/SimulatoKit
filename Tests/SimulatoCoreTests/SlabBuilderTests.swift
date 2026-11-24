//
//  SlabBuilderTests.swift
//  Role: the slab builder by truths and refusals that need no fixture (the abTEM pins are in SimulatoKitParityTests'
//        SlabPinsTests): every atom of the supercell is kept, the cell from lengths and angles is ASE's, planes stay off slice
//        boundaries, and each bad input is a typed refusal.
//

import Testing
import SimulatoCore

@Suite("Slab builder")
struct SlabBuilderTests {
    static let fcc: [CrystalCell.Site] = [SIMD3<Double>(0, 0, 0), SIMD3(0, 0.5, 0.5), SIMD3(0.5, 0, 0.5), SIMD3(0.5, 0.5, 0)]
        .map { CrystalCell.Site(atomicNumber: 13, fractional: $0) }

    func aluminium() throws -> CrystalCell {
        try CrystalCell(a: 4.05, b: 4.05, c: 4.05, alpha: 90, beta: 90, gamma: 90, sites: Self.fcc)
    }

    func magnesium() throws -> CrystalCell {
        try CrystalCell(a: 3.209, b: 3.209, c: 5.211, alpha: 90, beta: 90, gamma: 120, sites: [
            CrystalCell.Site(atomicNumber: 12, fractional: SIMD3(1.0 / 3, 2.0 / 3, 0.25)),
            CrystalCell.Site(atomicNumber: 12, fractional: SIMD3(2.0 / 3, 1.0 / 3, 0.75)),
        ])
    }

    /// abTEM 1.0.10 keeps 23 of the 24 atoms of Al [111] (ASE's cut drops one a rounding error below a face); the kit keeps
    /// all of them, none twice: 6 cells of 4 atoms, every pair at least the nearest-neighbour distance a / √2 apart.
    @Test func everyAtomOfTheSupercellIsKept() throws {
        for zone in [SIMD3(1, 1, 1), SIMD3(1, 1, 2), SIMD3(1, 1, 0), SIMD3(1, 2, 3)] {
            let unit = try SlabBuilder.orthogonalUnit(of: aluminium(), zoneAxis: zone, maximumRepetitions: 5)
            #expect(unit.structure.atoms.count == unit.cellCount * 4, "Al \(zone): \(unit.structure.atoms.count) atoms")
            #expect(abs(unit.structure.box.x * unit.structure.box.y * unit.structure.box.z
                        - Double(unit.cellCount) * 4.05 * 4.05 * 4.05) <= 1e-9, "Al \(zone): volume")
            let box = unit.structure.box
            var nearest = Double.infinity
            let atoms = unit.structure.atoms
            for i in atoms.indices {
                for j in atoms.indices where j > i {
                    var d = atoms[i].position - atoms[j].position
                    for a in 0..<3 { d[a] -= box[a] * (d[a] / box[a]).rounded() }
                    nearest = min(nearest, (d * d).sum().squareRoot())
                }
            }
            #expect(abs(nearest - 4.05 / 2.0.squareRoot()) <= 1e-9, "Al \(zone): nearest neighbours \(nearest) Å")
        }
        let al111 = try SlabBuilder.orthogonalUnit(of: aluminium(), zoneAxis: SIMD3(1, 1, 1), maximumRepetitions: 5)
        #expect(al111.structure.atoms.count == 24 && al111.cellCount == 6)
    }

    /// ASE's `cellpar_to_cell` for hexagonal Mg: b = (a cos 120°, a sin 120°, 0), c along z; 90° gives an exact zero.
    @Test func cellFromLengthsAndAnglesIsASEs() throws {
        let mg = try magnesium()
        #expect(mg.a == SIMD3(3.209, 0, 0))
        #expect(abs(mg.b.x + 3.209 / 2) <= 1e-15 && abs(mg.b.y - 3.209 * 3.0.squareRoot() / 2) <= 1e-15 && mg.b.z == 0)
        #expect(mg.c == SIMD3(0, 0, 5.211))
        let unit = try SlabBuilder.orthogonalUnit(of: mg, zoneAxis: SIMD3(0, 0, 1))
        #expect(unit.structure.atoms.count == 4 && unit.strain <= 1e-12, "Mg [0001]: the orthohexagonal cell")
        #expect(abs(unit.structure.box.y - 3.209 * 3.0.squareRoot()) <= 1e-12)
    }

    /// A slab's slices repeat with the period and no atom sits within a quarter slice of a boundary on Al [001] and [110].
    @Test func planesStayOffSliceBoundaries() throws {
        for zone in [SIMD3(0, 0, 1), SIMD3(1, 1, 0)] {
            let slab = try SlabBuilder.slab(of: aluminium(), zoneAxis: zone, thickness: 50, sliceThickness: 1)
            let h = slab.sliceThicknesses[0]
            #expect(slab.sliceThicknesses.count == Int((slab.period / 1).rounded(.up)) * slab.repeats.z)
            #expect(abs(slab.sliceThicknesses.reduce(0, +) - slab.structure.box.z) <= 1e-9)
            for atom in slab.structure.atoms {
                let q = atom.position.z / h
                #expect(abs(q - q.rounded()) * h >= h / 4 - 1e-9, "Al \(zone): z = \(atom.position.z) Å")
                #expect(atom.position.z >= 0 && atom.position.z < slab.structure.box.z)
            }
            #expect(abs(slab.structure.box.z - Double(slab.repeats.z) * slab.period) <= 1e-9)
            #expect(slab.repeats.z == Int((50 / slab.period).rounded()))
        }
    }

    @Test func badInputsAreTypedRefusals() throws {
        let al = try aluminium()
        #expect(throws: SimulatoError.invalidZoneAxis("[0 0 0] is no direction")) {
            try SlabBuilder.orthogonalUnit(of: al, zoneAxis: .zero)
        }
        #expect(throws: SimulatoError.invalidRepetitions(0)) {
            try SlabBuilder.orthogonalUnit(of: al, zoneAxis: SIMD3(0, 0, 1), maximumRepetitions: 0)
        }
        #expect(throws: SimulatoError.invalidRepetitions(65)) {
            try SlabBuilder.orthogonalUnit(of: al, zoneAxis: SIMD3(0, 0, 1), maximumRepetitions: 65)
        }
        #expect(throws: SimulatoError.invalidStrain(-1)) {
            try SlabBuilder.orthogonalUnit(of: al, zoneAxis: SIMD3(0, 0, 1), maximumStrain: -1)
        }
        let half = try CrystalCell(a: 4.05, b: 4.05, c: 4.05, alpha: 90, beta: 90, gamma: 90, sites: [
            CrystalCell.Site(atomicNumber: 13, fractional: .zero),
            CrystalCell.Site(atomicNumber: 12, fractional: SIMD3(0.5, 0.5, 0.5), occupancy: 0.4),
        ])
        #expect(throws: SimulatoError.partialOccupancy(site: 1, occupancy: 0.4)) {
            try SlabBuilder.slab(of: half, zoneAxis: SIMD3(0, 0, 1), thickness: 10)
        }
        // Mg [1 0 3] needs 11.3 % at 5 repetitions and 0.22 % at 10, 20 and 40: refused at 0.1 %, carrying the strain; accepted at 1 %.
        do {
            _ = try SlabBuilder.orthogonalUnit(of: magnesium(), zoneAxis: SIMD3(1, 0, 3))
            Issue.record("Mg [1 0 3] was not refused")
        } catch let SimulatoError.strainTooLarge(strain, maximum, _) {
            #expect(strain > maximum && maximum == 0.001 && strain < 0.003, "strain \(strain)")
        }
        #expect(try SlabBuilder.orthogonalUnit(of: magnesium(), zoneAxis: SIMD3(1, 0, 3), maximumStrain: 0.01).strain < 0.003)
        #expect(throws: SimulatoError.tooManyAtoms(count: 4 * 2 * 2 * 25, limit: 100)) {
            try SlabBuilder.slab(of: al, zoneAxis: SIMD3(0, 0, 1), lateralSize: SIMD2(8.1, 8.1), thickness: 101.25,
                                 maximumAtomCount: 100)
        }
        for t in [0.0, -1, .nan, .infinity] {
            #expect(throws: SimulatoError.self) { try SlabBuilder.slab(of: al, zoneAxis: SIMD3(0, 0, 1), thickness: t) }
            #expect(throws: SimulatoError.self) {
                try SlabBuilder.slab(of: al, zoneAxis: SIMD3(0, 0, 1), thickness: 10, sliceThickness: t)
            }
        }
        // Cells: flat, left-handed, non-finite, no sites, bad Z, bad occupancy, impossible angles.
        let x = SIMD3<Double>(4, 0, 0), y = SIMD3<Double>(0, 4, 0), z = SIMD3<Double>(0, 0, 4)
        let site = [CrystalCell.Site(atomicNumber: 13, fractional: .zero)]
        let bad: [() throws -> CrystalCell] = [
            { try CrystalCell(a: x, b: y, c: x + y, sites: site) },
            { try CrystalCell(a: y, b: x, c: z, sites: site) },
            { try CrystalCell(a: x, b: y, c: SIMD3(0, 0, .nan), sites: site) },
            { try CrystalCell(a: x, b: y, c: z, sites: []) },
            { try CrystalCell(a: x, b: y, c: z, sites: [CrystalCell.Site(atomicNumber: 0, fractional: .zero)]) },
            { try CrystalCell(a: x, b: y, c: z, sites: [CrystalCell.Site(atomicNumber: 13, fractional: .zero, occupancy: 0)]) },
            { try CrystalCell(a: x, b: y, c: z, sites: [CrystalCell.Site(atomicNumber: 13, fractional: SIMD3(.nan, 0, 0))]) },
            { try CrystalCell(a: 4, b: 4, c: 4, alpha: 10, beta: 10, gamma: 170, sites: site) },
            { try CrystalCell(a: 4, b: -4, c: 4, alpha: 90, beta: 90, gamma: 90, sites: site) },
        ]
        for (i, make) in bad.enumerated() {
            do {
                _ = try make()
                Issue.record("cell \(i) was not refused")
            } catch let error as SimulatoError {
                guard case .invalidCell = error else { Issue.record("cell \(i): \(error)"); continue }
            }
        }
    }

    /// The ladder cuts atoms only for the rung it takes (#130): its unit is the one cut at that rung directly, atom for atom.
    @Test func theLadderTakesTheRungItCuts() throws {
        for zone in [SIMD3(1, 1, 1), SIMD3(1, 2, 3), SIMD3(2, 1, 0)] {
            let taken = try SlabBuilder.orthogonalUnit(of: aluminium(), zoneAxis: zone)
            let direct = try SlabBuilder.orthogonalUnit(of: aluminium(), zoneAxis: zone, maximumRepetitions: taken.maximumRepetitions)
            #expect(taken.structure.box == direct.structure.box, "Al \(zone)")
            #expect(taken.structure.atoms.map(\.position) == direct.structure.atoms.map(\.position), "Al \(zone)")
            #expect(taken.supercell == direct.supercell && taken.strain == direct.strain, "Al \(zone)")
        }
    }

    /// Stop reaches the ladder (#130): a cancelled task is answered with `CancellationError` before any rung is cut.
    @Test func aCancelledLadderStops() async throws {
        let al = try aluminium()
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try SlabBuilder.orthogonalUnit(of: al, zoneAxis: SIMD3(1, 2, 3))
        }
        await #expect(throws: CancellationError.self) { try await task.value }
    }

    /// A model's slab is the crystal's slab (#121): Al's conventional cell as a model, the beam along z, gives the slab of Al down
    /// [0 0 1] (the zone frame is the identity there) at every thickness and lateral size, atom for atom; with no thickness it is
    /// one period (a padded model is never stacked); a bad box is refused the same way.
    @Test func aModelsSlabIsTheCrystalsSlab() throws {
        let a = 4.05
        let model = try CrystalCell(cellRows: [SIMD3(a, 0, 0), SIMD3(0, a, 0), SIMD3(0, 0, a)], atomicNumbers: [13, 13, 13, 13],
                                    positions: Self.fcc.map { $0.fractional * a })
        for (lateral, thickness) in [(nil, 20.0), (SIMD2(12.0, 9.0), 41.0)] as [(SIMD2<Double>?, Double)] {
            let fromModel = try SlabBuilder.slab(ofModel: model, lateralSize: lateral, thickness: thickness)
            let fromCrystal = try SlabBuilder.slab(of: aluminium(), zoneAxis: SIMD3(0, 0, 1), lateralSize: lateral, thickness: thickness)
            #expect(fromModel.repeats == fromCrystal.repeats)
            #expect(fromModel.sliceThicknesses == fromCrystal.sliceThicknesses)
            #expect(fromModel.structure.atoms.count == fromCrystal.structure.atoms.count)
            #expect(fromModel.structure.box == fromCrystal.structure.box)
            #expect(fromModel.zShift == fromCrystal.zShift)
            func key(_ a: AtomicStructure.Atom) -> [Double] { [a.position.z, a.position.y, a.position.x].map { ($0 * 1e6).rounded() } }
            let m = fromModel.structure.atoms.map(key).sorted { $0.lexicographicallyPrecedes($1) }
            let c = fromCrystal.structure.atoms.map(key).sorted { $0.lexicographicallyPrecedes($1) }
            #expect(m == c, "the cut and the shift are the crystal's")
        }
        let onePeriod = try SlabBuilder.slab(ofModel: model)
        #expect(onePeriod.repeats == SIMD3(1, 1, 1))
        #expect(throws: SimulatoError.self) { try SlabBuilder.slab(ofModel: model, thickness: 0) }
        #expect(throws: SimulatoError.self) { try SlabBuilder.slab(ofModel: model, sliceThickness: -1) }
    }
}

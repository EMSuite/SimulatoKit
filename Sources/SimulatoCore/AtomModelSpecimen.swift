//
//  AtomModelSpecimen.swift
//  Role: a specimen from an atom model (an extended XYZ or POSCAR file, as DiffractoKit's `AtomModel` reads it): the
//        model as a `CrystalCell` of every atom, padded with vacuum as ASE's `Atoms.center(vacuum:axis:)` when it has no
//        cell, and turned so the beam runs along x, y or z (abTEM 1.0.10 `orthogonalize_cell(plane:)` with the planes
//        "yz", "zx" and "xy") before the slab builder's orthogonal unit. The kit takes plain data (ADR 0005: no subject
//        kit imports another): atomic numbers, Cartesian positions [Å] and the cell's rows.
//

import Foundation

extension CrystalCell {
    /// A model as a cell: every atom is a site, its fractional coordinates p · cell⁻¹ (wrapped into the cell by the slab
    /// builder).
    /// - Throws: `SimulatoError.invalidCell` for counts that differ, a non-finite position and the refusals of
    ///   ``init(a:b:c:sites:)`` (a left-handed or flat cell, no atoms, Z outside 1 ... 118, an occupancy outside (0, 1]).
    public init(cellRows: [SIMD3<Double>], atomicNumbers: [Int], positions: [SIMD3<Double>],
                occupancies: [Double]? = nil) throws {
        guard cellRows.count == 3 else { throw SimulatoError.invalidCell("a cell has three rows, got \(cellRows.count)") }
        guard positions.count == atomicNumbers.count, occupancies.map({ $0.count == positions.count }) ?? true else {
            throw SimulatoError.invalidCell("\(atomicNumbers.count) atomic numbers, \(positions.count) positions and \(occupancies?.count ?? positions.count) occupancies")
        }
        guard let inverse = Lattice.inverse(cellRows) else { throw SimulatoError.invalidCell("the cell spans no volume") }
        var sites: [Site] = []
        sites.reserveCapacity(positions.count)
        for (i, p) in positions.enumerated() {
            guard p.x.isFinite, p.y.isFinite, p.z.isFinite else {
                throw SimulatoError.invalidCell("atom \(i) has a non-finite position")
            }
            sites.append(Site(atomicNumber: atomicNumbers[i], fractional: Lattice.vecmat(p, inverse),
                              occupancy: occupancies?[i] ?? 1))
        }
        try self.init(a: cellRows[0], b: cellRows[1], c: cellRows[2], sites: sites)
    }

    /// A model padded with vacuum, as ASE 3.29.0 `Atoms.center(vacuum:axis:)` on a cell that is none or rectangular along
    /// x, y and z: along every padded axis the cell becomes the atoms' extent plus twice `vacuum` and the atoms move so
    /// that the nearest lies `vacuum` from the face; the other axes keep the given side and positions.
    /// - Parameters:
    ///   - box: the rectangular cell's sides [Å], or nil for a model without a cell (every axis must then be padded).
    ///   - axes: the padded axes, 0 (x), 1 (y), 2 (z).
    /// - Throws: `SimulatoError.invalidCell` for a vacuum that is not finite and non-negative, an axis outside 0 ... 2,
    ///   a model without a cell not padded on every axis, a side that is not positive, and the refusals of
    ///   ``init(cellRows:atomicNumbers:positions:occupancies:)``.
    public static func padded(atomicNumbers: [Int], positions: [SIMD3<Double>], box: SIMD3<Double>?, vacuum: Double,
                              axes: Set<Int> = [0, 1, 2], occupancies: [Double]? = nil) throws -> CrystalCell {
        guard vacuum.isFinite, vacuum >= 0 else { throw SimulatoError.invalidCell("vacuum must be finite and ≥ 0, got \(vacuum) Å") }
        guard axes.isSubset(of: [0, 1, 2]) else { throw SimulatoError.invalidCell("axes must lie in 0 ... 2, got \(axes.sorted())") }
        guard box != nil || axes == [0, 1, 2] else {
            throw SimulatoError.invalidCell("a model without a cell is padded on every axis")
        }
        guard !positions.isEmpty else { throw SimulatoError.invalidCell("the model has no atoms") }
        if let b = box, !(b.x > 0 && b.y > 0 && b.z > 0 && b.x.isFinite && b.y.isFinite && b.z.isFinite) {
            throw SimulatoError.invalidCell("the cell's sides must be finite and positive, got (\(b.x), \(b.y), \(b.z)) Å")
        }
        var side = box ?? SIMD3(1, 1, 1)  // ASE's Cell.complete() of no cell: unit vectors
        var moved = positions
        for i in axes.sorted() {
            let along = positions.map { $0[i] }
            let p0 = along.min()!, p1 = along.max()!
            // ASE: lng = (p1 − p0 + 2 vacuum) − height; shift = ((lng + height − p1) − p0) / 2; the side grows by lng.
            let height = side[i]
            let longer = (p1 - p0 + 2 * vacuum) - height
            let shift = 0.5 * ((longer + height - p1) - p0)
            side[i] = height * (1 + longer / height)
            for k in moved.indices { moved[k][i] += shift }
        }
        guard side.x > 0, side.y > 0, side.z > 0, side.x.isFinite, side.y.isFinite, side.z.isFinite else {
            throw SimulatoError.invalidCell("the padded cell has sides (\(side.x), \(side.y), \(side.z)) Å; give some vacuum")
        }
        return try CrystalCell(cellRows: [SIMD3(side.x, 0, 0), SIMD3(0, side.y, 0), SIMD3(0, 0, side.z)],
                               atomicNumbers: atomicNumbers, positions: moved, occupancies: occupancies)
    }
}

extension SlabBuilder {
    /// The Cartesian axis of a model the beam runs along.
    public enum BeamAxis: Sendable, Hashable, CaseIterable {
        /// abTEM's plane "yz": the box's x, y and z are the model's y, z and x.
        case x
        /// abTEM's plane "zx": the box's x, y and z are the model's z, x and y.
        case y
        /// abTEM's plane "xy" (the default): the model as it is.
        case z

        /// The rows of the model-to-box rotation, a cyclic permutation (proper, so the model keeps its handedness).
        var rotation: [SIMD3<Double>] {
            switch self {
            case .x: return [SIMD3(0, 1, 0), SIMD3(0, 0, 1), SIMD3(1, 0, 0)]
            case .y: return [SIMD3(0, 0, 1), SIMD3(1, 0, 0), SIMD3(0, 1, 0)]
            case .z: return [SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, 1)]
            }
        }
    }

    /// The orthogonal unit of an atom model with the beam along a Cartesian axis (abTEM's
    /// `orthogonalize_cell(plane:)`): the model's axes permuted, then the fewest repetitions of ``repetitionLadder`` whose
    /// strain is at most `maximumStrain`. With the beam along z a model whose cell is rectangular along x, y and z passes
    /// through unchanged apart from the wrap and the 1e−6 Å snap to the faces (along x or y its axes are permuted);
    /// another cell is cut into the best orthogonal cell with its strain. The orientation is the permutation.
    ///
    /// The beam enters at the box's z = 0 and travels +z (SimulatoKit#126): a box z is depth below the entrance face, the face
    /// nearest the source. With `.z`, the model's atom at the smallest z (after the wrap) is the first the beam meets; to enter
    /// a model from its other face, turn it 180° about x before building (a mirror would change its handedness).
    /// - Throws: `SimulatoError.strainTooLarge` with the least strain found; `invalidStrain`; `partialOccupancy` for a
    ///   partly occupied atom; `invalidCell` when no orthogonal cell is found.
    public static func orthogonalUnit(ofModel model: CrystalCell, beamAlong axis: BeamAxis = .z,
                                      maximumStrain: Double = defaultMaximumStrain) throws -> OrthogonalUnit {
        guard maximumStrain.isFinite, maximumStrain >= 0 else { throw SimulatoError.invalidStrain(maximumStrain) }
        try checkOccupancy(model)
        return try ladder(model, rotation: axis.rotation, maximumStrain: maximumStrain)
    }

    /// A slab of an atom model (``orthogonalUnit(ofModel:beamAlong:maximumStrain:)``), as
    /// ``slab(of:zoneAxis:lateralSize:thickness:sliceThickness:maximumStrain:maximumAtomCount:allowsPartialOccupancy:)`` makes one of a crystal: the unit
    /// repeated to at least `lateralSize` in x and y (one unit when nil), along the beam to the whole number of periods nearest
    /// `thickness` (at least one), or one period when nil, cut in equal slices of at most `sliceThickness` per period, shifted
    /// along z so no atomic plane sits on a slice boundary. Box z is depth below the entrance face.
    ///
    /// Give `thickness` only for a model that is periodic along the beam (a cell of a crystal or a bicrystal). A model padded with
    /// vacuum (``CrystalCell/padded(atomicNumbers:positions:box:vacuum:axes:occupancies:)``, a particle) is one period as it
    /// stands: repeating it would stack copies with their vacuum between them.
    /// - Throws: `SimulatoError.invalidBox` for a lateral size or thickness that is not finite and positive;
    ///   `invalidSliceThickness`; `tooManyAtoms` above `maximumAtomCount`; the refusals of the unit.
    public static func slab(ofModel model: CrystalCell, beamAlong axis: BeamAxis = .z, lateralSize: SIMD2<Double>? = nil,
                            thickness: Double? = nil, sliceThickness: Double = 1,
                            maximumStrain: Double = defaultMaximumStrain,
                            maximumAtomCount: Int = defaultMaximumAtomCount) throws -> Slab {
        let lateral = lateralSize ?? SIMD2(0, 0)
        guard thickness.map({ $0.isFinite && $0 > 0 }) ?? true, lateral.x.isFinite, lateral.y.isFinite, lateral.x >= 0, lateral.y >= 0 else {
            throw SimulatoError.invalidBox(lateral.x, lateral.y, thickness ?? .nan)
        }
        guard sliceThickness.isFinite, sliceThickness > 0 else { throw SimulatoError.invalidSliceThickness(sliceThickness) }
        let unit = try orthogonalUnit(ofModel: model, beamAlong: axis, maximumStrain: maximumStrain)
        return try slab(of: unit, lateral: lateral, thickness: thickness ?? unit.structure.box.z, sliceThickness: sliceThickness,
                        maximumAtomCount: maximumAtomCount)
    }
}

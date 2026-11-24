//
//  SlabBuilder.swift
//  Role: a crystal cell turned to a zone axis and made orthogonal (a port of abTEM 1.0.10 `orthogonalize_cell` with
//        `best_orthogonal_cell` and `decompose_affine_transform`), and the slab built from it: repeated to a lateral size and
//        a thickness, cut in slices that repeat with the crystal, and shifted along the beam so no atomic plane sits on a
//        slice boundary. The result is the `AtomicStructure` the multislice takes.
//
//  Method (docs/parity.md, "Slab building"):
//    1. Turn the cell: z along the zone axis [u v w] (pointing at the source, the suite's orientation convention), x along
//       the shortest lattice row at right angles to it (the one closest to py4DSTEM's x reference on a tie; py4DSTEM's
//       reference itself when no row is at right angles), y = z × x.
//    2. abTEM's `best_orthogonal_cell`: of every n₁a + n₂b + n₃c with |nᵢ| ≤ the repetitions, the shortest one most nearly
//       along each axis; the box is the column norms of those three vectors.
//    3. abTEM's cut: the supercell matrix V = round(box · cell⁻¹) and the atoms inside the cell V · cell. abTEM cuts with
//       ASE's `cut`, whose `% 1.0` can push an atom a rounding error below a face onto the far face and drop it (Al and
//       Cu [111] lose 1 of 24 atoms in 1.0.10); the kit takes the new cell's atoms by lattice arithmetic (fractional
//       coordinates within 1e−9 of an integer are that integer), so every one of det(V) × the cell's atoms is kept.
//    4. abTEM's affine transform A = (V · cell)⁻¹ · box puts the new cell on the box; its scale and shear
//       (`decompose_affine_transform`) are the strain the unit carries. Coordinates within 1e−6 Å of a face are set on it.
//    5. The slab turns z into depth (#126): the unit's z points at the source, while the multislice transmits from z = 0
//       along +z as abTEM does, so the slab mirrors z → period − z in every unit. Mirroring crystal and beam together
//       leaves every exit wave's (x, y) as it was, so images and patterns stay as seen from the source.
//

import simd

/// The orthogonal repeat unit of a crystal along a zone axis.
public struct OrthogonalUnit: Sendable, Hashable {
    /// The unit: its box and atoms in Cartesian coordinates [Å], the zone axis along +z, pointing at the source (a `Slab`
    /// mirrors it to depth for the multislice).
    public let structure: AtomicStructure
    /// The rows of the integer matrix V: box x, y and z as n₁a + n₂b + n₃c. det V cells make the unit.
    public let supercell: [SIMD3<Int>]
    /// The largest |nᵢ| the search allowed (abTEM's `max_repetitions`).
    public let maximumRepetitions: Int
    /// The normal stretch of the affine transform along x, y and z (1 is none), as abTEM's `decompose_affine_transform`.
    public let scale: SIMD3<Double>
    /// The shear of the affine transform: xy, xz and yz.
    public let shear: SIMD3<Double>
    /// The crystal's orientation in the box, in the suite's convention (DiffractoKit's `CrystalOrientation.matrix`,
    /// py4DSTEM's `Orientation.matrix`): the columns are the box's x, y and z axes written in the frame of the cell's
    /// vectors. The third column is the zone axis, pointing at the source; the beam travels along −z.
    public let orientation: simd_double3x3

    /// The largest of |scale − 1| and |shear| (0.001 is 0.1 %).
    public var strain: Double {
        var worst = 0.0
        for a in 0..<3 {
            let stretch: Double = abs(abs(scale[a]) - 1)
            let skew: Double = abs(shear[a])
            worst = max(worst, stretch, skew)
        }
        return worst
    }

    /// The number of cells in the unit, det V.
    public var cellCount: Int {
        let r0: SIMD3<Int> = supercell[0], r1: SIMD3<Int> = supercell[1], r2: SIMD3<Int> = supercell[2]
        let m0: Int = r1.y * r2.z - r1.z * r2.y
        let m1: Int = r1.x * r2.z - r1.z * r2.x
        let m2: Int = r1.x * r2.y - r1.y * r2.x
        return r0.x * m0 - r0.y * m1 + r0.z * m2
    }

    public static func == (l: OrthogonalUnit, r: OrthogonalUnit) -> Bool {
        guard l.structure == r.structure, l.supercell == r.supercell, l.maximumRepetitions == r.maximumRepetitions else {
            return false
        }
        guard l.scale == r.scale, l.shear == r.shear else { return false }
        let a = l.orientation, b = r.orientation
        return a.columns.0 == b.columns.0 && a.columns.1 == b.columns.1 && a.columns.2 == b.columns.2
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(structure)
        hasher.combine(supercell)
        hasher.combine(maximumRepetitions)
        hasher.combine(scale)
        hasher.combine(shear)
        hasher.combine(orientation.columns.0)
        hasher.combine(orientation.columns.1)
        hasher.combine(orientation.columns.2)
    }
}

/// A slab of crystal along a zone axis, ready for the multislice.
public struct Slab: Sendable, Hashable {
    /// The slab: the unit shifted along z by ``zShift``, wrapped into its box, mirrored to depth (z → period − z) and
    /// repeated. z is the depth below the entrance face, the face nearest the source: the multislice meets z = 0 first.
    public let structure: AtomicStructure
    /// The orthogonal unit it repeats (before the shift and the mirror; its z points at the source).
    public let unit: OrthogonalUnit
    /// How many units along x, y and z.
    public let repeats: SIMD3<Int>
    /// The slices to cut it in: k equal slices per period (k = ⌈period / requested thickness⌉), the same in every period.
    /// Pass them to `Potential(structure:nx:ny:sliceThicknesses:)`.
    public let sliceThicknesses: [Double]
    /// How far the atoms were moved along the unit's z [Å], within one slice, before the mirror: the shift that keeps every
    /// atomic plane farthest from a slice boundary (the mirror keeps that distance, as the boundaries repeat with the period).
    public let zShift: Double

    /// The crystal's repeat along the beam [Å].
    public var period: Double { unit.structure.box.z }
    /// The distance from the nearest atom to the nearest slice boundary [Å] (half a slice at best).
    public var planeMargin: Double { SlabBuilder.planeMargin(unit.structure.atoms.map(\.position.z), shift: zShift,
                                                             sliceHeight: sliceThicknesses[0]) }
    /// The crystal's orientation in the box (``OrthogonalUnit/orientation``).
    public var orientation: simd_double3x3 { unit.orientation }
    /// The strain the unit carries (``OrthogonalUnit/strain``).
    public var strain: Double { unit.strain }
}

/// Builds the orthogonal unit and the slab of a crystal along a zone axis.
public enum SlabBuilder {
    /// The strain a unit may carry by default: 0.1 %, the precision an indexed pattern resolves.
    public static let defaultMaximumStrain = 0.001
    /// The repetitions tried in turn until the strain holds.
    public static let repetitionLadder = [5, 10, 20, 40]
    /// The largest repetitions one search may use ((2n + 1)³ lattice vectors are tried).
    public static let maximumRepetitionsLimit = 64
    /// The most atoms a slab may have by default.
    public static let defaultMaximumAtomCount = 2_000_000
    /// The lattice rows searched for the x axis: |nᵢ| ≤ 10.
    static let rowSearch = 10
    /// The z shifts tried: j h / 720 for j = 0 ... 719, h the slice height.
    static let shiftSteps = 720
    /// The most atoms one orthogonal unit may have.
    public static let unitAtomLimit = 10_000_000

    /// The orthogonal unit at a fixed number of repetitions (abTEM's `orthogonalize_cell(max_repetitions:)`), whatever its
    /// strain.
    /// - Throws: `SimulatoError.invalidZoneAxis` for [0 0 0]; `invalidRepetitions` outside 1 ... 64; `partialOccupancy` for
    ///   a site with occupancy below 1; `invalidCell` when no orthogonal cell is found within the repetitions.
    public static func orthogonalUnit(of cell: CrystalCell, zoneAxis: SIMD3<Int>,
                                      maximumRepetitions: Int) throws -> OrthogonalUnit {
        guard (1...maximumRepetitionsLimit).contains(maximumRepetitions) else {
            throw SimulatoError.invalidRepetitions(maximumRepetitions)
        }
        try checkOccupancy(cell)
        let rotation = try zoneRotation(cell, zoneAxis: zoneAxis)
        return try unit(cell, rotation: rotation, repetitions: maximumRepetitions)
    }

    /// The orthogonal unit with the fewest repetitions of ``repetitionLadder`` whose strain is at most `maximumStrain`.
    /// - Throws: `SimulatoError.strainTooLarge` with the least strain found (and its repetitions) when no rung holds it;
    ///   `invalidStrain` for a limit that is not finite and non-negative; the refusals of
    ///   ``orthogonalUnit(of:zoneAxis:maximumRepetitions:)``.
    public static func orthogonalUnit(of cell: CrystalCell, zoneAxis: SIMD3<Int>,
                                      maximumStrain: Double = defaultMaximumStrain) throws -> OrthogonalUnit {
        guard maximumStrain.isFinite, maximumStrain >= 0 else { throw SimulatoError.invalidStrain(maximumStrain) }
        try checkOccupancy(cell)
        let rotation = try zoneRotation(cell, zoneAxis: zoneAxis)
        var best: OrthogonalUnit?
        for n in repetitionLadder {
            let u = try unit(cell, rotation: rotation, repetitions: n)
            if u.strain <= maximumStrain { return u }
            if best == nil || u.strain < best!.strain { best = u }
        }
        throw SimulatoError.strainTooLarge(strain: best!.strain, maximum: maximumStrain, repetitions: best!.maximumRepetitions)
    }

    /// A slab: the orthogonal unit (``orthogonalUnit(of:zoneAxis:maximumStrain:)``) repeated to at least `lateralSize` in x
    /// and y (one unit when nil) and to the whole number of periods nearest `thickness` (at least one), cut in k equal
    /// slices per period of at most `sliceThickness`, with the atoms shifted along z so no atomic plane sits on a slice
    /// boundary.
    /// - Throws: `SimulatoError.invalidBox` for a lateral size or thickness that is not finite and positive;
    ///   `invalidSliceThickness`; `tooManyAtoms` above `maximumAtomCount`; the refusals of the unit.
    public static func slab(of cell: CrystalCell, zoneAxis: SIMD3<Int>, lateralSize: SIMD2<Double>? = nil,
                            thickness: Double, sliceThickness: Double = 1,
                            maximumStrain: Double = defaultMaximumStrain,
                            maximumAtomCount: Int = defaultMaximumAtomCount) throws -> Slab {
        let lateral = lateralSize ?? SIMD2(0, 0)
        guard thickness.isFinite, thickness > 0, lateral.x.isFinite, lateral.y.isFinite, lateral.x >= 0, lateral.y >= 0 else {
            throw SimulatoError.invalidBox(lateral.x, lateral.y, thickness)
        }
        guard sliceThickness.isFinite, sliceThickness > 0 else { throw SimulatoError.invalidSliceThickness(sliceThickness) }
        let unit = try orthogonalUnit(of: cell, zoneAxis: zoneAxis, maximumStrain: maximumStrain)
        let box = unit.structure.box
        func count(_ want: Double, _ side: Double) -> Double { max(1, (want / side - 1e-9).rounded(.up)) }
        let n = SIMD3(count(lateral.x, box.x), count(lateral.y, box.y),
                      max(1, (thickness / box.z).rounded(.toNearestOrEven)))
        let perPeriod = (box.z / sliceThickness).rounded(.up)
        let total = Double(unit.structure.atoms.count) * n.x * n.y * n.z
        guard total <= Double(maximumAtomCount) else {
            throw SimulatoError.tooManyAtoms(count: total < 9e18 ? Int(total) : .max, limit: maximumAtomCount)
        }
        guard perPeriod * n.z <= 1_000_000 else { throw SimulatoError.invalidSliceThickness(sliceThickness) }
        let k = Int(perPeriod), repeats = SIMD3(Int(n.x), Int(n.y), Int(n.z))
        let h = box.z / Double(k)
        let z = unit.structure.atoms.map(\.position.z)
        let shift = zShift(z, sliceHeight: h)
        func intoBox(_ z: Double) -> Double {
            let w = z.truncatingRemainder(dividingBy: box.z)
            return w < 0 ? w + box.z : w
        }
        let shifted = unit.structure.atoms.map { atom -> AtomicStructure.Atom in
            var p = atom.position
            p.z = intoBox(box.z - intoBox(p.z + shift))  // shifted, then mirrored to depth below the entrance face
            return AtomicStructure.Atom(atomicNumber: atom.atomicNumber, position: p)
        }
        let structure = try AtomicStructure(box: box, atoms: shifted).repeated(repeats.x, repeats.y, repeats.z)
        return Slab(structure: structure, unit: unit, repeats: repeats,
                    sliceThicknesses: [Double](repeating: h, count: k * repeats.z), zShift: shift)
    }

    // MARK: The steps

    static func checkOccupancy(_ cell: CrystalCell) throws {
        for (i, s) in cell.sites.enumerated() where s.occupancy < 1 - 1e-6 {
            throw SimulatoError.partialOccupancy(site: i, occupancy: s.occupancy)
        }
    }

    /// The rows ex, ey, ez of the crystal-to-box rotation (p_box = R p_crystal): ez along the zone axis, ex along the
    /// shortest lattice row at right angles to it (|r · ez| ≤ 1e−9 |r|, lengths equal within 1e−9 relative); on a tie the
    /// row closest to py4DSTEM's x reference, then the larger (n₁, n₂, n₃); py4DSTEM's reference when no row qualifies.
    static func zoneRotation(_ cell: CrystalCell, zoneAxis uvw: SIMD3<Int>) throws -> [SIMD3<Double>] {
        guard uvw != .zero else { throw SimulatoError.invalidZoneAxis("[0 0 0] is no direction") }
        let rows = cell.rows
        let d = Lattice.vecmat(SIMD3(Double(uvw.x), Double(uvw.y), Double(uvw.z)), rows)
        let length = Lattice.length(d)
        guard length.isFinite, length > 0 else { throw SimulatoError.invalidZoneAxis("[\(uvw.x) \(uvw.y) \(uvw.z)] has no length") }
        let ez = d / length
        let xref = py4DSTEMReference(ez)
        var best: (length: Double, cosine: Double, n: SIMD3<Int>, r: SIMD3<Double>)?
        let m = rowSearch
        for i in -m...m {
            for j in -m...m {
                for k in -m...m where !(i == 0 && j == 0 && k == 0) {
                    let r = Lattice.vecmat(SIMD3(Double(i), Double(j), Double(k)), rows)
                    let l = Lattice.length(r)
                    guard abs(Lattice.dot(r, ez)) <= 1e-9 * l else { continue }
                    let c = Lattice.dot(r, xref) / l
                    let n = SIMD3<Int>(i, j, k)
                    guard let b = best else {
                        best = (length: l, cosine: c, n: n, r: r)
                        continue
                    }
                    let shorter = l < b.length * (1 - 1e-9)
                    let tie = l <= b.length * (1 + 1e-9)
                    let larger: Bool = i != b.n.x ? i > b.n.x : (j != b.n.y ? j > b.n.y : k > b.n.z)
                    let closer: Bool = c > b.cosine + 1e-9
                    let level: Bool = c >= b.cosine - 1e-9
                    if shorter || (tie && (closer || (level && larger))) {
                        best = (length: l, cosine: c, n: n, r: r)
                    }
                }
            }
        }
        let ex = best.map { $0.r / $0.length } ?? xref
        var ey = Lattice.cross(ez, ex)
        ey /= Lattice.length(ey)
        return [ex, ey, ez]
    }

    /// py4DSTEM's x reference for a beam direction (DiffractoKit's `CrystalOrientation.zoneAxisFrame`): ŷ × z when
    /// |z_z| > 1 − 1e−6, else (0, 0, −1), made perpendicular to z.
    static func py4DSTEMReference(_ ez: SIMD3<Double>) -> SIMD3<Double> {
        var x = abs(ez.z) > 1 - 1e-6 ? Lattice.cross(SIMD3(0, 1, 0), ez) : SIMD3(0, 0, -1)
        let y = Lattice.cross(ez, x)
        x = Lattice.cross(y, ez)
        return x / Lattice.length(x)
    }

    /// abTEM's `best_orthogonal_cell` (eps 1e−12): the box sides, the column norms of the three chosen vectors.
    static func bestOrthogonalBox(_ cell: [SIMD3<Double>], repetitions r: Int) -> SIMD3<Double> {
        let eps = 1e-12
        let side = 2 * r + 1
        var indices: [SIMD3<Int>] = []
        var vectors: [SIMD3<Double>] = []
        var norms: [Double] = []
        indices.reserveCapacity(side * side * side)
        vectors.reserveCapacity(side * side * side)
        norms.reserveCapacity(side * side * side)
        for nx in -r...r {  // NumPy's C order: nx outermost, nz innermost
            for ny in -r...r {
                for nz in -r...r {
                    let v = Lattice.vecmat(SIMD3(Double(nx), Double(ny), Double(nz)), cell)
                    let a = SIMD3(abs(v.x), abs(v.y), abs(v.z))
                    indices.append(SIMD3(nx, ny, nz))
                    vectors.append(a)
                    norms.append(Lattice.length(a))
                }
            }
        }
        var chosen: [SIMD3<Double>] = []
        for axis in 0..<3 {
            var angles = [Double](repeating: 0, count: vectors.count)
            var largest = -Double.infinity
            for q in vectors.indices {
                angles[q] = vectors[q][axis] / (norms[q] > eps ? norms[q] : eps)
                largest = max(largest, angles[q])
            }
            var pick = -1
            for q in vectors.indices where largest - angles[q] < eps && norms[q] > eps {
                if pick < 0 || norms[q] < norms[pick] { pick = q }
            }
            let n = indices[pick]
            let component = Lattice.vecmat(SIMD3(Double(n.x), Double(n.y), Double(n.z)), cell)[axis]
            let sign: Double = component > 0 ? 1 : (component < 0 ? -1 : 0)
            chosen.append(Lattice.vecmat(sign * SIMD3(Double(n.x), Double(n.y), Double(n.z)), cell))
        }
        var box = SIMD3<Double>(0, 0, 0)
        for v in chosen { box += v * v }
        return SIMD3<Double>(box.x.squareRoot(), box.y.squareRoot(), box.z.squareRoot())
    }

    /// The orthogonal unit of the turned cell at `repetitions`.
    static func unit(_ cell: CrystalCell, rotation: [SIMD3<Double>], repetitions: Int) throws -> OrthogonalUnit {
        let rt = Lattice.transpose(rotation)
        // abTEM: the turned cell with every |component| below 1e-6 set to zero.
        let turned = cell.rows.map { v -> SIMD3<Double> in
            let w = Lattice.vecmat(v, rt)
            return SIMD3(abs(w.x) < 1e-6 ? 0 : w.x, abs(w.y) < 1e-6 ? 0 : w.y, abs(w.z) < 1e-6 ? 0 : w.z)
        }
        let box = bestOrthogonalBox(turned, repetitions: repetitions)
        let d: [SIMD3<Double>] = [SIMD3(box.x, 0, 0), SIMD3(0, box.y, 0), SIMD3(0, 0, box.z)]
        let orthogonal = turned[0].y == 0 && turned[0].z == 0 && turned[1].x == 0 && turned[1].z == 0
            && turned[2].x == 0 && turned[2].y == 0
        let v: [SIMD3<Double>]
        let unchanged = orthogonal && turned[0].x == box.x && turned[1].y == box.y && turned[2].z == box.z
        if unchanged {  // abTEM's early return: the cell as it is, no transform
            v = [SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, 1)]
        } else {
            guard let inv = Lattice.inverse(turned) else { throw SimulatoError.invalidCell("the turned cell has no inverse") }
            v = Lattice.matmul(d, inv).map {
                SIMD3($0.x.rounded(.toNearestOrEven), $0.y.rounded(.toNearestOrEven), $0.z.rounded(.toNearestOrEven))
            }
        }
        let cells = Lattice.determinant(v).rounded(.toNearestOrEven)
        guard cells >= 1, cells <= 1e7, box.x > 0, box.y > 0, box.z > 0,
              let vInverse = Lattice.inverse(v) else {
            throw SimulatoError.invalidCell("no orthogonal cell within \(repetitions) repetitions")
        }
        let newCell = Lattice.matmul(v, turned)
        guard let newInverse = Lattice.inverse(newCell) else {
            throw SimulatoError.invalidCell("no orthogonal cell within \(repetitions) repetitions")
        }
        let identity: [SIMD3<Double>] = [SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, 1)]
        let affine: [SIMD3<Double>] = unchanged ? identity : Lattice.matmul(newInverse, d)

        // The atoms of the new cell: every site in every cell translation that covers it, kept when its fractional
        // coordinates in the new cell lie in [0, 1) (within 1e-9 of an integer counts as that integer).
        var corners: [SIMD3<Double>] = []
        for c0 in [0.0, 1.0] {
            for c1 in [0.0, 1.0] {
                for c2 in [0.0, 1.0] { corners.append(Lattice.vecmat(SIMD3<Double>(c0, c1, c2), v)) }
            }
        }
        var low = corners[0], high = corners[0]
        for corner in corners {
            low = pointwiseMin(low, corner)
            high = pointwiseMax(high, corner)
        }
        let lo = SIMD3<Int>(Int(low.x.rounded(.down)), Int(low.y.rounded(.down)), Int(low.z.rounded(.down)))
        let hi = SIMD3<Int>(Int(high.x.rounded(.up)), Int(high.y.rounded(.up)), Int(high.z.rounded(.up)))
        guard cells * Double(cell.sites.count) <= Double(unitAtomLimit) else {
            throw SimulatoError.tooManyAtoms(count: Int(cells) * cell.sites.count, limit: unitAtomLimit)
        }
        let expected = Int(cells) * cell.sites.count
        var atoms: [AtomicStructure.Atom] = []
        atoms.reserveCapacity(expected)
        for site in cell.sites {
            for t0 in lo.x...hi.x {
                for t1 in lo.y...hi.y {
                    for t2 in lo.z...hi.z {
                        let f = wrapped(site.fractional) + SIMD3(Double(t0), Double(t1), Double(t2))
                        var s = Lattice.vecmat(f, vInverse)
                        for a in 0..<3 {
                            let r = s[a].rounded(.toNearestOrEven)
                            if abs(s[a] - r) <= 1e-9 { s[a] = r }
                        }
                        guard s.x >= 0, s.x < 1, s.y >= 0, s.y < 1, s.z >= 0, s.z < 1 else { continue }
                        var p = s * box
                        for a in 0..<3 where p[a] > box[a] - 1e-6 || p[a] < 1e-6 { p[a] = 0 }
                        atoms.append(AtomicStructure.Atom(atomicNumber: site.atomicNumber, position: p))
                    }
                }
            }
        }
        guard atoms.count == expected else {
            throw SimulatoError.invalidCell("the cut kept \(atoms.count) of \(expected) atoms")
        }
        let (scale, shear) = decompose(affine)
        // The columns are the rows of R: the box's axes written in the crystal's frame.
        let orientation = simd_double3x3(columns: (rotation[0], rotation[1], rotation[2]))
        return OrthogonalUnit(structure: try AtomicStructure(box: box, atoms: atoms),
                              supercell: v.map { SIMD3(Int($0.x), Int($0.y), Int($0.z)) },
                              maximumRepetitions: repetitions, scale: scale, shear: shear, orientation: orientation)
    }

    /// A fractional position wrapped into [0, 1) (within 1e−9 of 1 counts as 0).
    static func wrapped(_ f: SIMD3<Double>) -> SIMD3<Double> {
        var w = f
        for a in 0..<3 {
            w[a] -= w[a].rounded(.down)
            if w[a] >= 1 - 1e-9 { w[a] = 0 }
        }
        return w
    }

    /// abTEM's `decompose_affine_transform`: ZS = chol(AᵀA)ᵀ, scale = diag ZS, shear = (ZS₀₁/s₀, ZS₀₂/s₀, ZS₁₂/s₁); the
    /// x scale changes sign when A turns the handedness.
    static func decompose(_ a: [SIMD3<Double>]) -> (scale: SIMD3<Double>, shear: SIMD3<Double>) {
        let m = Lattice.matmul(Lattice.transpose(a), a)
        var l = [SIMD3<Double>](repeating: .zero, count: 3)  // lower Cholesky factor, rows
        for i in 0..<3 {
            for j in 0...i {
                var s = m[i][j]
                for k in 0..<j { s -= l[i][k] * l[j][k] }
                if i == j { l[i][j] = s.squareRoot() } else { l[i][j] = s / l[j][j] }
            }
        }
        var scale = SIMD3(l[0][0], l[1][1], l[2][2])
        let shear = SIMD3(l[1][0] / scale.x, l[2][0] / scale.x, l[2][1] / scale.y)
        if Lattice.determinant(a) < 0 { scale.x = -scale.x }
        return (scale, shear)
    }

    /// The z shift of j h / 720 (j = 0 ... 719) that keeps the planes farthest from a slice boundary (distances rounded to
    /// 1e−9 Å, the smallest j on a tie).
    static func zShift(_ z: [Double], sliceHeight h: Double) -> Double {
        var bestJ = 0, bestF = -1.0
        for j in 0..<shiftSteps {
            let delta = Double(j) * h / Double(shiftSteps)
            let f = (planeMargin(z, shift: delta, sliceHeight: h) * 1e9).rounded(.toNearestOrEven) / 1e9
            if f > bestF {
                bestJ = j
                bestF = f
            }
        }
        return Double(bestJ) * h / Double(shiftSteps)
    }

    /// The smallest distance of z + shift from a multiple of h.
    static func planeMargin(_ z: [Double], shift: Double, sliceHeight h: Double) -> Double {
        var f = Double.infinity
        for value in z {
            let q = (value + shift) / h
            f = min(f, abs(q - (q + 0.5).rounded(.down)) * h)
        }
        return f
    }
}

/// 3 × 3 arithmetic on rows, as NumPy writes it (v @ M is a row vector times a matrix).
enum Lattice {
    static func dot(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> Double { a.x * b.x + a.y * b.y + a.z * b.z }
    static func cross(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> SIMD3<Double> {
        SIMD3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
    }
    static func length(_ a: SIMD3<Double>) -> Double { dot(a, a).squareRoot() }
    static func vecmat(_ v: SIMD3<Double>, _ m: [SIMD3<Double>]) -> SIMD3<Double> { v.x * m[0] + v.y * m[1] + v.z * m[2] }
    static func matmul(_ a: [SIMD3<Double>], _ b: [SIMD3<Double>]) -> [SIMD3<Double>] { a.map { vecmat($0, b) } }
    static func transpose(_ m: [SIMD3<Double>]) -> [SIMD3<Double>] {
        [SIMD3(m[0].x, m[1].x, m[2].x), SIMD3(m[0].y, m[1].y, m[2].y), SIMD3(m[0].z, m[1].z, m[2].z)]
    }
    static func determinant(_ m: [SIMD3<Double>]) -> Double { dot(m[0], cross(m[1], m[2])) }
    /// The inverse by cofactors, nil for a singular or non-finite matrix.
    static func inverse(_ m: [SIMD3<Double>]) -> [SIMD3<Double>]? {
        let d = determinant(m)
        guard d.isFinite, d != 0 else { return nil }
        let c0 = cross(m[1], m[2]), c1 = cross(m[2], m[0]), c2 = cross(m[0], m[1])
        return [SIMD3(c0.x, c1.x, c2.x) / d, SIMD3(c0.y, c1.y, c2.y) / d, SIMD3(c0.z, c1.z, c2.z) / d]
    }
}

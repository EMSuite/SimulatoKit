//
//  AtomicStructure.swift
//  Role: the atoms to simulate: an orthogonal box and atoms with Cartesian positions in Å.
//        A minimal local type on purpose: the kit stays standalone (docs/adr/0001-standalone-structure-types.md).
//

/// An orthogonal box with atoms in Cartesian coordinates [Å], the beam along z.
public struct AtomicStructure: Sendable, Hashable {
    public struct Atom: Sendable, Hashable {
        /// Atomic number Z.
        public var atomicNumber: Int
        /// Cartesian position (x, y, z) [Å].
        public var position: SIMD3<Double>
        /// The fraction of cells in which the site holds this atom, in (0, 1]. Atoms on one site share it: their
        /// occupancies add up to at most 1. A partly occupied atom is drawn per configuration (`FrozenPhonons`); a
        /// potential takes whole atoms only.
        public var occupancy: Double

        public init(atomicNumber: Int, position: SIMD3<Double>, occupancy: Double = 1) {
            self.atomicNumber = atomicNumber
            self.position = position
            self.occupancy = occupancy
        }
    }

    /// The box's sides (Lx, Ly, Lz) [Å]; the cell is diagonal.
    public let box: SIMD3<Double>
    public let atoms: [Atom]

    /// - Throws: `SimulatoError.invalidBox` for a non-finite or non-positive side, `invalidPosition` for a non-finite
    ///   coordinate, `invalidOccupancy` for an occupancy outside (0, 1]. Atoms outside the box are wrapped into it when the potential is built (periodic boundaries).
    public init(box: SIMD3<Double>, atoms: [Atom]) throws {
        guard box.x.isFinite, box.y.isFinite, box.z.isFinite, box.x > 0, box.y > 0, box.z > 0 else {
            throw SimulatoError.invalidBox(box.x, box.y, box.z)
        }
        for (i, a) in atoms.enumerated()
        where !(a.position.x.isFinite && a.position.y.isFinite && a.position.z.isFinite) {
            throw SimulatoError.invalidPosition(index: i)
        }
        if let i = atoms.firstIndex(where: { !($0.occupancy > 0 && $0.occupancy <= 1) }) {
            throw SimulatoError.invalidOccupancy(atom: i, occupancy: atoms[i].occupancy)
        }
        self.box = box
        self.atoms = atoms
    }

    /// Whether any atom is only partly occupied.
    public var hasPartialOccupancy: Bool { atoms.contains { $0.occupancy < 1 } }

    /// The structure repeated nx × ny × nz times, in ASE's order (x outermost, then y, then z, the atoms innermost).
    /// - Throws: `invalidShape` for a repeat count below 1, `tooManyAtoms` when the atoms' count overflows an `Int`,
    ///   `invalidBox` when a side of the repeated box is not finite.
    public func repeated(_ nx: Int, _ ny: Int, _ nz: Int) throws -> AtomicStructure {
        guard nx >= 1, ny >= 1, nz >= 1 else {
            throw SimulatoError.invalidShape("repeat counts must be at least 1, got (\(nx), \(ny), \(nz))")
        }
        var total = atoms.count, overflow = false
        for n in [nx, ny, nz] {
            let r = total.multipliedReportingOverflow(by: n)
            total = r.partialValue
            overflow = overflow || r.overflow
        }
        guard !overflow else { throw SimulatoError.tooManyAtoms(count: .max, limit: .max) }
        let side = box * SIMD3(Double(nx), Double(ny), Double(nz))
        guard side.x.isFinite, side.y.isFinite, side.z.isFinite else { throw SimulatoError.invalidBox(side.x, side.y, side.z) }
        var out: [Atom] = []
        out.reserveCapacity(total)
        guard !atoms.isEmpty else { return try AtomicStructure(box: side, atoms: []) }
        for mx in 0..<nx {
            for my in 0..<ny {
                for mz in 0..<nz {
                    let shift = SIMD3<Double>(Double(mx) * box.x, Double(my) * box.y, Double(mz) * box.z)
                    for var a in atoms {
                        a.position += shift
                        out.append(a)
                    }
                }
            }
        }
        return try AtomicStructure(box: side, atoms: out)
    }
}

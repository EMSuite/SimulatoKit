//
//  CrystalCell.swift
//  Role: a crystal as a general cell, three lattice vectors and the sites of one cell in fractional coordinates (ASE's
//        `Atoms` with a general cell): the input of the slab builder. A minimal local type on purpose: the kit stays
//        standalone (docs/adr/0001-standalone-structure-types.md), and an app maps a phase's cell and expanded sites into it.
//

import Foundation

/// A crystal: three lattice vectors in Å and every site of one cell in fractional coordinates (the space group already
/// applied, as ASE's `crystal` gives them).
///
/// The vectors are written in the crystal's Cartesian frame; the suite's frame (DiffractoKit's `UnitCell`, ASE's
/// `cellpar_to_cell`) puts a along x and b in the xy plane, which ``init(a:b:c:alpha:beta:gamma:sites:)`` builds. The
/// slab builder's orientation is written in the same frame as the vectors.
public struct CrystalCell: Sendable, Hashable {
    /// One atom of the cell.
    public struct Site: Sendable, Hashable {
        /// Atomic number Z.
        public var atomicNumber: Int
        /// Fractional coordinates along a, b and c.
        public var fractional: SIMD3<Double>
        /// The fraction of cells in which the site holds this atom, in (0, 1]. The slab builder refuses a partly occupied
        /// site.
        public var occupancy: Double

        public init(atomicNumber: Int, fractional: SIMD3<Double>, occupancy: Double = 1) {
            self.atomicNumber = atomicNumber
            self.fractional = fractional
            self.occupancy = occupancy
        }
    }

    /// The lattice vectors a, b and c [Å].
    public let a: SIMD3<Double>
    public let b: SIMD3<Double>
    public let c: SIMD3<Double>
    public let sites: [Site]

    /// A cell from its lattice vectors.
    /// - Throws: `SimulatoError.invalidCell` for a non-finite vector, vectors that are left-handed or span (almost) no
    ///   volume (a · (b × c) at most 1e−6 of |a| |b| |c|), no sites, a non-finite fractional coordinate, a Z outside
    ///   1 ... 118, or an occupancy outside (0, 1].
    public init(a: SIMD3<Double>, b: SIMD3<Double>, c: SIMD3<Double>, sites: [Site]) throws {
        for v in [a, b, c] where !(v.x.isFinite && v.y.isFinite && v.z.isFinite) {
            throw SimulatoError.invalidCell("a lattice vector is not finite")
        }
        let volume = Lattice.dot(a, Lattice.cross(b, c))
        let scale = Lattice.length(a) * Lattice.length(b) * Lattice.length(c)
        guard volume.isFinite, scale.isFinite, volume > 1e-6 * scale else {
            throw SimulatoError.invalidCell("the lattice vectors are left-handed or span no volume (a · (b × c) = \(volume) Å³)")
        }
        guard !sites.isEmpty else { throw SimulatoError.invalidCell("the cell has no sites") }
        for (i, s) in sites.enumerated() {
            guard s.fractional.x.isFinite, s.fractional.y.isFinite, s.fractional.z.isFinite else {
                throw SimulatoError.invalidCell("site \(i) has a non-finite fractional coordinate")
            }
            guard (1...118).contains(s.atomicNumber) else {
                throw SimulatoError.invalidCell("site \(i) has Z = \(s.atomicNumber), outside 1 ... 118")
            }
            guard s.occupancy.isFinite, s.occupancy > 0, s.occupancy <= 1 else {
                throw SimulatoError.invalidCell("site \(i) has occupancy \(s.occupancy), outside (0, 1]")
            }
        }
        self.a = a
        self.b = b
        self.c = c
        self.sites = sites
    }

    /// A cell from its lengths [Å] and angles [degrees] in the suite's frame: a along x, b in the xy plane, c completing
    /// a right-handed set (ASE 3.29.0 `cellpar_to_cell` with its defaults, including its exact zero cosine within
    /// 2 ulp of 90°).
    /// - Throws: `SimulatoError.invalidCell` for a non-finite or non-positive length, angles that make no cell, and the
    ///   refusals of ``init(a:b:c:sites:)``.
    public init(a: Double, b: Double, c: Double, alpha: Double, beta: Double, gamma: Double, sites: [Site]) throws {
        for v in [a, b, c] where !(v.isFinite && v > 0) {
            throw SimulatoError.invalidCell("cell lengths must be finite and positive, got (\(a), \(b), \(c)) Å")
        }
        for v in [alpha, beta, gamma] where !(v.isFinite && v > 0 && v < 180) {
            throw SimulatoError.invalidCell("cell angles must lie in (0, 180)°, got (\(alpha), \(beta), \(gamma))°")
        }
        let eps = 2 * (90.0).ulp  // NumPy's 2 * spacing(90.0)
        let cosAlpha = abs(alpha - 90) < eps ? 0 : cos(alpha * Double.pi / 180)
        let cosBeta = abs(beta - 90) < eps ? 0 : cos(beta * Double.pi / 180)
        let cosGamma: Double, sinGamma: Double
        if abs(gamma - 90) < eps {
            cosGamma = 0
            sinGamma = 1
        } else {
            cosGamma = cos(gamma * Double.pi / 180)
            sinGamma = sin(gamma * Double.pi / 180)
        }
        let cx = cosBeta
        let cy = (cosAlpha - cosBeta * cosGamma) / sinGamma
        let czSquared = 1 - cx * cx - cy * cy
        guard czSquared > 0 else {
            throw SimulatoError.invalidCell("the angles (\(alpha), \(beta), \(gamma))° make no cell")
        }
        try self.init(a: SIMD3(a, 0, 0), b: SIMD3(b * cosGamma, b * sinGamma, 0),
                      c: SIMD3(c * cx, c * cy, c * czSquared.squareRoot()), sites: sites)
    }

    /// The rows a, b, c.
    var rows: [SIMD3<Double>] { [a, b, c] }

    /// The cell's volume [Å³].
    public var volume: Double { Lattice.dot(a, Lattice.cross(b, c)) }
}

//
//  OccupancyDraw.swift
//  Role: partly occupied sites drawn per configuration. Atoms within `siteTolerance` of each other (after the wrap into
//        the box) share a site; each configuration draws one uniform number per site and keeps at most one of its atoms,
//        atom i with probability equal to its occupancy, or none. abTEM has no partial occupancy, so the reference is
//        the closed form: the potential is linear in the atoms, so the mean over configurations is the
//        occupancy-weighted potential (docs/parity.md, "Partly occupied sites"). Sites are grouped within the box only:
//        two atoms on either side of a periodic face are not one site (a crystal's sites sit at the same coordinates).
//

import SimulatoCore

/// The partly occupied sites of a structure and the seeded draw that fills them per configuration.
public struct OccupancyDraw: Sendable, Hashable {
    /// Atoms closer than this [Å] share a site.
    public static let siteTolerance = 1e-4

    /// The atoms of every site that holds a partly occupied atom (indices into the structure's atoms, ascending), the
    /// sites in the order of their first atom.
    public let sites: [[Int]]
    /// The occupancy of every atom of every site, in the order of `sites`.
    let occupancies: [[Double]]
    /// The structure's atom count.
    let atomCount: Int

    /// The shared sites of `structure`.
    /// - Throws: `SimulatoError.invalidOccupancy` for a site whose atoms' occupancies add up to more than 1 (by more than
    ///   1e−6), naming the site's first atom and its total.
    public init(structure: AtomicStructure) throws {
        let atoms = Slicing.wrapped(structure)
        atomCount = atoms.count
        let partial = atoms.indices.filter { atoms[$0].occupancy < 1 }
        guard !partial.isEmpty else {
            sites = []
            occupancies = []
            return
        }
        let cell = 10 * Self.siteTolerance
        func key(_ p: SIMD3<Double>) -> SIMD3<Int> {
            SIMD3(Int((p.x / cell).rounded(.down)), Int((p.y / cell).rounded(.down)), Int((p.z / cell).rounded(.down)))
        }
        var buckets: [SIMD3<Int>: [Int]] = [:]
        for (i, a) in atoms.enumerated() { buckets[key(a.position), default: []].append(i) }

        // Union–find over the partly occupied atoms and every atom within the tolerance of one.
        var parent = Dictionary(uniqueKeysWithValues: partial.map { ($0, $0) })
        func root(_ i: Int) -> Int {
            var r = i
            while let p = parent[r], p != r { r = p }
            return r
        }
        for i in partial {
            let k = key(atoms[i].position)
            for dx in -1...1 {
                for dy in -1...1 {
                    for dz in -1...1 {
                        for j in buckets[k &+ SIMD3(dx, dy, dz)] ?? [] where j != i {
                            let d = atoms[j].position - atoms[i].position
                            guard (d * d).sum() <= Self.siteTolerance * Self.siteTolerance else { continue }
                            if parent[j] == nil { parent[j] = j }
                            let (ri, rj) = (root(i), root(j))
                            if ri != rj { parent[max(ri, rj)] = min(ri, rj) }
                        }
                    }
                }
            }
        }
        var members: [Int: [Int]] = [:]
        for i in parent.keys { members[root(i), default: []].append(i) }
        let grouped = members.values.map { $0.sorted() }.sorted { $0[0] < $1[0] }
        for site in grouped {
            let total = site.reduce(0) { $0 + atoms[$1].occupancy }
            guard total <= 1 + 1e-6 else { throw SimulatoError.invalidOccupancy(atom: site[0], occupancy: total) }
        }
        sites = grouped
        occupancies = grouped.map { $0.map { atoms[$0].occupancy } }
    }

    /// Whether every atom is whole (nothing to draw).
    public var isEmpty: Bool { sites.isEmpty }

    /// The atoms configuration `configuration` keeps: for every site one uniform u in [0, 1) from the kit's generator
    /// seeded with `FrozenPhonons.configurationSeed(seed ^ salt, configuration)`, and the first of its atoms whose
    /// cumulative occupancy exceeds u, or none. Atoms on no partly occupied site are always kept.
    public func kept(seed: UInt64, configuration: Int) -> [Bool] {
        var keep = [Bool](repeating: true, count: atomCount)
        var g = GaussianGenerator(seed: FrozenPhonons.configurationSeed(seed ^ Self.salt, configuration))
        for (site, occupancy) in zip(sites, occupancies) {
            let u = Double(g.nextUInt64() >> 11) * 0x1p-53
            var cumulative = 0.0
            var chosen: Int?
            for (i, o) in zip(site, occupancy) {
                cumulative += o
                if chosen == nil, u < cumulative { chosen = i }
                keep[i] = false
            }
            if let c = chosen { keep[c] = true }
        }
        return keep
    }

    /// Keeps the draws apart from the displacements of the same seed and configuration.
    static let salt: UInt64 = 0x6F63_6375_7061_6E63  // "occupanc"
}

//
//  Potential.swift
//  Role: abTEM's lazy `Potential` (`potentials/iam.py`): a structure cut into slices on a grid, the wrapped atoms
//        with their slice and the parameterisation, without the arrays. `build()` gives the `ProjectedPotential`
//        (abTEM's `PotentialArray`) on the CPU; on `.metal` the multislice builds each slice on the GPU instead.
//

import SimulatoCore

/// A structure sliced on a grid, ready to be built into a projected potential (on the CPU by `build()`, or slice by slice
/// on the GPU by the `.metal` backend). As in abTEM, the beam enters at z = 0 and travels along +z: the first slice is the
/// lowest z (a `Slab`'s structure is already depth below its entrance face).
public struct Potential: Sendable, Hashable {
    public let grid: Grid
    /// The thickness of every slice [Å].
    public let sliceThicknesses: [Double]
    public let parameterization: Parameterization
    /// The atoms wrapped into the box, in the structure's atom order.
    public let atoms: [AtomicStructure.Atom]
    /// The slice of each atom, in the same order.
    public let sliceOfAtom: [Int]

    /// Slices of `sliceThickness` (abTEM's rule: ⌈Lz / s⌉ equal slices), at most `maximumSliceCount`.
    public init(structure: AtomicStructure, nx: Int, ny: Int, sliceThickness: Double,
                parameterization: Parameterization = .standard,
                maximumSliceCount: Int = Slicing.maximumSliceCount) throws {
        let t = try Slicing.thicknesses(depth: structure.box.z, requested: sliceThickness, maximumCount: maximumSliceCount)
        try self.init(structure: structure, nx: nx, ny: ny, sliceThicknesses: t, parameterization: parameterization,
                      maximumSliceCount: maximumSliceCount)
    }

    /// The grid from a target `sampling` [Å] (abTEM's `Potential(sampling=)`): ⌈box / sampling⌉ points per axis.
    public init(structure: AtomicStructure, sampling: Double, sliceThickness: Double,
                parameterization: Parameterization = .standard,
                maximumSliceCount: Int = Slicing.maximumSliceCount) throws {
        try self.init(structure: structure, sampling: (sampling, sampling), sliceThickness: sliceThickness,
                      parameterization: parameterization, maximumSliceCount: maximumSliceCount)
    }

    /// The same with a sampling per axis (x, y).
    public init(structure: AtomicStructure, sampling: (x: Double, y: Double), sliceThickness: Double,
                parameterization: Parameterization = .standard,
                maximumSliceCount: Int = Slicing.maximumSliceCount) throws {
        let grid = try Grid(extentX: structure.box.x, extentY: structure.box.y, samplingX: sampling.x, samplingY: sampling.y)
        try self.init(structure: structure, nx: grid.nx, ny: grid.ny, sliceThickness: sliceThickness,
                      parameterization: parameterization, maximumSliceCount: maximumSliceCount)
    }

    /// The given slice thicknesses (they must sum to the box's depth), at most `maximumSliceCount`.
    public init(structure: AtomicStructure, nx: Int, ny: Int, sliceThicknesses: [Double],
                parameterization: Parameterization = .standard,
                maximumSliceCount: Int = Slicing.maximumSliceCount) throws {
        let grid = try Grid(nx: nx, ny: ny, extentX: structure.box.x, extentY: structure.box.y)
        let thicknesses = try Slicing.validated(sliceThicknesses, depth: structure.box.z, maximumCount: maximumSliceCount)
        for a in structure.atoms where !parameterization.supports(atomicNumber: a.atomicNumber) {
            throw SimulatoError.unsupportedElement(a.atomicNumber)
        }
        if let i = structure.atoms.firstIndex(where: { $0.occupancy < 1 }) {
            throw SimulatoError.partlyOccupiedAtom(index: i, occupancy: structure.atoms[i].occupancy)
        }
        let atoms = Slicing.wrapped(structure)
        self.grid = grid
        self.sliceThicknesses = thicknesses
        self.parameterization = parameterization
        self.atoms = atoms
        self.sliceOfAtom = try Slicing.sliceIndices(z: atoms.map(\.position.z), thicknesses: thicknesses)
    }

    /// The number of slices.
    public var sliceCount: Int { sliceThicknesses.count }

    /// The total thickness [Å].
    public var thickness: Double { sliceThicknesses.reduce(0, +) }

    /// The atoms of slice j, in the structure's order.
    public func atoms(inSlice j: Int) -> [AtomicStructure.Atom] {
        atoms.indices.filter { sliceOfAtom[$0] == j }.map { atoms[$0] }
    }

    /// The sum of the slices as an image [V·Å] (abTEM `Potential.project`, which builds the potential first): `build()`
    /// then `ProjectedPotential.project()`.
    public func project() throws -> RealField {
        try build().project()
    }

    /// The atoms of every slice, in the structure's order, sorted in one pass (`atoms(inSlice:)` for each slice).
    package func atomsBySlice() -> [[AtomicStructure.Atom]] {
        var out = [[AtomicStructure.Atom]](repeating: [], count: sliceCount)
        for (a, j) in zip(atoms, sliceOfAtom) where out.indices.contains(j) { out[j].append(a) }
        return out
    }

    /// The projected potential of every slice, on the CPU in float64. A slice that repeats an earlier one
    /// (`firstEqualSlices()`) shares its values, bit for bit what building it would give (#122). Cancellation is checked
    /// before each slice (`CancellationError`).
    public func build() throws -> ProjectedPotential {
        let builder = try SliceBuilder(grid: grid, parameterization: parameterization)
        var slices: [PotentialSlice] = []
        slices.reserveCapacity(sliceThicknesses.count)
        // The atoms sorted into their slices once (filtering every atom for every slice was atoms x slices, #61).
        let bySlice = atomsBySlice()
        let first = firstEqualSlices(bySlice)
        for (j, dz) in sliceThicknesses.enumerated() {
            try Task.checkCancellation()
            let values: [Double]
            if first[j] == j {
                WorkCounter.current?.countSlicePotential()
                values = try builder.potential(of: bySlice[j])
            } else {
                values = slices[first[j]].values
            }
            slices.append(PotentialSlice(thickness: dz, values: values))
        }
        return ProjectedPotential(grid: grid, slices: slices, sliceOfAtom: sliceOfAtom, firstEqualSlices: first)
    }

    /// For each slice, the first slice with the same atoms: the same atomic numbers at the same x and y, in the same
    /// order. The infinite projection reads only those, so equal slices have equal potentials and transmission functions
    /// bit for bit (a crystal slab repeats its slices every period; a frozen-phonon configuration repeats none).
    package func firstEqualSlices() -> [Int] { firstEqualSlices(atomsBySlice()) }

    func firstEqualSlices(_ bySlice: [[AtomicStructure.Atom]]) -> [Int] {
        var seen: [[SIMD3<Double>]: Int] = [:]
        return bySlice.enumerated().map { (j: Int, sliceAtoms: [AtomicStructure.Atom]) -> Int in
            let key = sliceAtoms.map { SIMD3(Double($0.atomicNumber), $0.position.x, $0.position.y) }
            if let k = seen[key] { return k }
            seen[key] = j
            return j
        }
    }
}

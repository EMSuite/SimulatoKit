//
//  AtomsBySliceTests.swift
//  Role: the CPU potential sorts its atoms into slices and elements in one pass (#61): the grouping equals filtering
//        each slice, every slice and element keeps the structure's order (the pinned potentials guard the built values).
//

import Testing
import SimulatoCore
import SimulatoKitTestSupport
import SimulatoMultislice

@Suite("Atoms sorted into slices once")
struct AtomsBySliceTests {
    @Test func theGroupingEqualsFilteringEachSlice() throws {
        // Al with Mg and Si mixed in, over several slices, in a scrambled z order.
        let al = try aluminiumStructure(cells: (2, 2, 3))
        let atoms = al.atoms.enumerated().map { k, a in
            AtomicStructure.Atom(atomicNumber: [13, 12, 14][k % 3], position: a.position)
        }.reversed()
        let structure = try AtomicStructure(box: al.box, atoms: Array(atoms))
        let potential = try Potential(structure: structure, nx: 24, ny: 24, sliceThickness: 1)
        #expect(potential.sliceCount > 4)
        let grouped = potential.atomsBySlice()
        #expect(grouped.count == potential.sliceCount)
        for j in 0..<potential.sliceCount {
            #expect(grouped[j] == potential.atoms(inSlice: j))
        }
        #expect(grouped.reduce(0) { $0 + $1.count } == potential.atoms.count)
        // build() goes through the grouping (the pinned potentials guard its values).
        let built = try potential.build()
        #expect(built.slices.count == potential.sliceCount)
    }
}

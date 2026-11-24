//
//  EntranceFaceTests.swift
//  Role: the multislice meets a slab's face nearest the source first (#126): a gold layer at the top of the cell along
//        [0 0 1] lands in the first slice of the potential, and in the last along [0 0 −1].
//

import Testing
import SimulatoCore
import SimulatoMultislice

@Suite("Entrance face")
struct EntranceFaceTests {
    @Test func goldAtTheSourceSideIsInTheFirstSlice() throws {
        let layered = try CrystalCell(a: 4, b: 4, c: 10, alpha: 90, beta: 90, gamma: 90, sites: [
            CrystalCell.Site(atomicNumber: 79, fractional: SIMD3(0, 0, 0.9)),
            CrystalCell.Site(atomicNumber: 6, fractional: SIMD3(0.5, 0.5, 0.1)),
        ])
        let cases: [(zone: SIMD3<Int>, goldFirst: Bool)] = [(SIMD3(0, 0, 1), true), (SIMD3(0, 0, -1), false)]
        for (zone, goldFirst) in cases {
            let slab = try SlabBuilder.slab(of: layered, zoneAxis: zone, thickness: 10, sliceThickness: 1)
            let potential = try Potential(structure: slab.structure, nx: 16, ny: 16, sliceThicknesses: slab.sliceThicknesses)
            let last: Int = potential.sliceThicknesses.count - 1
            let gold: Int = potential.sliceOfAtom[potential.atoms.firstIndex(where: { $0.atomicNumber == 79 })!]
            let carbon: Int = potential.sliceOfAtom[potential.atoms.firstIndex(where: { $0.atomicNumber == 6 })!]
            #expect(gold == (goldFirst ? 0 : last), "\(zone): gold in slice \(gold) of \(last + 1)")
            #expect(carbon == (goldFirst ? last : 0), "\(zone): carbon in slice \(carbon) of \(last + 1)")
        }
    }
}

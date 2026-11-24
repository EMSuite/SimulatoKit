//
//  TransmissionCacheTests.swift
//  Role: slices that repeat are built once (#122): a slab's potential and transmissions are made once per distinct
//        slice and equal a one-period slab's, slice for slice, bit for bit on the CPU; on Metal the cached run stays
//        within the registered tolerance of the uncached one; the plan says when it caches.
//

import Testing
import SimulatoCore
import SimulatoMultislice
import SimulatoKitTestSupport

@Suite("Transmission cache")
struct TransmissionCacheTests {
    static let fcc: [CrystalCell.Site] = [SIMD3<Double>(0, 0, 0), SIMD3(0, 0.5, 0.5), SIMD3(0.5, 0, 0.5), SIMD3(0.5, 0.5, 0)]
        .map { CrystalCell.Site(atomicNumber: 13, fractional: $0) }

    /// Al [001] on 32², slices of at most 1 Å (5 per period), `periods` deep.
    static func slab(periods: Int) throws -> (Potential, perPeriod: Int) {
        let al = try CrystalCell(a: 4.05, b: 4.05, c: 4.05, alpha: 90, beta: 90, gamma: 90, sites: fcc)
        let slab = try SlabBuilder.slab(of: al, zoneAxis: SIMD3(0, 0, 1), lateralSize: SIMD2(8.1, 8.1),
                                        thickness: 4.05 * Double(periods), sliceThickness: 1)
        let potential = try Potential(structure: slab.structure, nx: 32, ny: 32, sliceThicknesses: slab.sliceThicknesses)
        return (potential, slab.sliceThicknesses.count / slab.repeats.z)
    }

    @Test func aSlabBuildsEachDistinctSliceOnceAndEqualsOnePeriod() throws {
        let (deep, k) = try Self.slab(periods: 10)
        let (one, _) = try Self.slab(periods: 1)
        let first = deep.firstEqualSlices()
        #expect(first.count == 10 * k)
        for j in k..<first.count { #expect(first[j] == first[j - k], "slice \(j) repeats slice \(j - k)") }
        let distinct = Set(first).count
        #expect(distinct <= k)

        let beam = try ElectronBeam(energy: 200e3)
        let counter = WorkCounter()
        let (built, transmissions) = try WorkCounter.$current.withValue(counter) {
            () throws -> (ProjectedPotential, [ComplexField]) in
            let built = try deep.build()
            let t = try Multislice.transmissionFunctions(built, beam: beam)
            return (built, t)
        }
        #expect(counter.slicePotentialCount == distinct, "\(counter.slicePotentialCount) slices built, \(distinct) distinct")
        #expect(counter.transmissionCount == distinct, "\(counter.transmissionCount) transmissions, \(distinct) distinct")

        let reference = try one.build()
        let referenceT = try Multislice.transmissionFunctions(reference, beam: beam)
        for j in 0..<built.slices.count {
            #expect(built.slices[j].values == reference.slices[j % k].values, "V of slice \(j)")
            #expect(transmissions[j] == referenceT[j % k], "t of slice \(j)")
        }
    }

    @Test(.enabled(if: Multislice.metalIsAvailable, "no Metal device"))
    func metalCachedEqualsUncachedWithinTheRegisteredTolerance() throws {
        let (deep, _) = try Self.slab(periods: 10)
        let beam = try ElectronBeam(energy: 200e3)
        let wave = Waves.planeWave(beam: beam, grid: deep.grid)
        let counter = WorkCounter()
        let cached = try WorkCounter.$current.withValue(counter) { () throws -> [Waves] in
            try Multislice.run(wave, through: deep, exitPlanes: .every(10), backend: .metal)
        }
        #expect(counter.transmissionCount == Set(deep.firstEqualSlices()).count, "one transmission per distinct slice")
        let uncached = try Multislice.run(wave, through: deep, exitPlanes: .every(10), backend: .metal,
                                          options: ExecutionOptions(metalTransmissionCacheBytes: 0))
        let cpu = try Multislice.run(wave, through: deep, exitPlanes: .every(10))
        #expect(cached.count == uncached.count)
        for (a, b) in zip(cached, uncached) { #expect(relMax(a.field, b.field) <= MetalAccuracy.registered.wave) }
        for (a, b) in zip(cached, cpu) { #expect(relMax(a.field, b.field) <= MetalAccuracy.registered.wave) }
        // The same through built slices.
        let built = try deep.build()
        let cachedBuilt = try Multislice.run(wave, through: built, exitPlanes: .every(10), backend: .metal)
        for (a, b) in zip(cachedBuilt, cpu) { #expect(relMax(a.field, b.field) <= MetalAccuracy.registered.wave) }
    }

    /// 250 slices at 1024², 8 distinct: cached at 64 MiB; not when every slice differs or the cache is too small.
    @Test func thePlanSaysWhenItCaches() throws {
        func plan(_ distinct: Int?, cacheBytes: Int = ExecutionOptions.standard.metalTransmissionCacheBytes) throws
            -> SimulationPlan {
            try SimulationPlan(SimulationPlan.Request(
                extent: SIMD2(51.2, 51.2), thickness: 250, energy: 200e3, largestAngle: 20, sampling: 0.05,
                distinctSlices: distinct, transmissionCacheBytes: cacheBytes), limits: SimulationLimits(memoryBudget: .max, largestBuffer: .max))
        }
        let cached = try plan(8)
        #expect(cached.grid.nx == 1024 && cached.sliceCount == 250)
        #expect(cached.cachesTransmissions && cached.memory.transmissionCache == 8 * 1024 * 1024 * 8)
        let every = try plan(nil)
        #expect(!every.cachesTransmissions && every.memory.transmissionCache == 0)
        #expect(cached.memory.total - every.memory.total == cached.memory.transmissionCache)
        let small = try plan(8, cacheBytes: 8 * 1024 * 1024 * 8 - 1)
        #expect(!small.cachesTransmissions && small.memory.transmissionCache == 0)
    }
}

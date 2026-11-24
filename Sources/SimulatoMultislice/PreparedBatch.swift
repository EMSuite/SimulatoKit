//
//  PreparedBatch.swift
//  Role: what a batch of waves reuses: the potential made ready once (transmission functions or GPU slices), the CPU
//        workers' FFT plans and the propagators, so a scan run in chunks of positions prepares once (#61). The tests
//        count the preparations through `WorkCounter` (SimulatoCore).
//

import SimulatoCore

/// A potential made ready once for many batches of one energy (and, on `.metal`, one tilt). Made by
/// `Multislice.prepareBatch`, run by `Multislice.runBatch(_:prepared:options:progress:)`.
package struct PreparedBatch {
    enum Storage {
        case cpu(ProjectedPotential, [ComplexField], CPUWorkspace)
        case metal(MetalPreparedPotential)
    }

    let storage: Storage
    let grid: Grid
    let energy: Double
    let tilt: SIMD2<Double>
}

/// The FFT plans (one per worker thread) and the propagators that CPU rounds reuse. One calling thread uses it at a
/// time; within a round, plan k belongs to worker k alone.
package final class CPUWorkspace: @unchecked Sendable {
    struct PropagatorKey: Hashable {
        let energy: Double
        let tilt: SIMD2<Double>
        let thickness: Double
    }

    let grid: Grid
    private var plans: [FFT2D] = []
    private var propagators: [PropagatorKey: ComplexField] = [:]

    package init(grid: Grid) {
        self.grid = grid
    }

    /// Makes sure `count` plans exist (on the calling thread, before a round).
    func reservePlans(_ count: Int) throws {
        while plans.count < count {
            plans.append(try FFT2D(nx: grid.nx, ny: grid.ny))
            WorkCounter.current?.countFFTPlan()
        }
    }

    /// Worker k's plan (`reservePlans` first).
    func plan(_ k: Int) -> FFT2D { plans[k] }

    /// The propagator of each slice thickness for `wave`'s energy and tilt, built once (on the calling thread).
    func propagators(for wave: Waves, thicknesses: Set<Double>) -> [Double: ComplexField] {
        var out: [Double: ComplexField] = [:]
        for dz in thicknesses {
            let key = PropagatorKey(energy: wave.beam.energy, tilt: wave.tilt, thickness: dz)
            if let p = propagators[key] {
                out[dz] = p
            } else {
                let p = Multislice.propagator(grid: grid, beam: wave.beam, thickness: dz, tilt: wave.tilt)
                propagators[key] = p
                out[dz] = p
            }
        }
        return out
    }
}

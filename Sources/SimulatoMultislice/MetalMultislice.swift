//
//  MetalMultislice.swift
//  Role: the backends' entry points: `.metal` runs on the GPU through `MetalEngine.swift` (float32, the potential
//        and transmissions on the device); `.cpu` batches run positions on at most 8 threads, bit for bit equal to
//        sequential runs.
//

import Foundation
import Metal
import Synchronization
import SimulatoCore

enum MetalMultislice {
    /// Whether a Metal device exists (the tests skip the Metal path with a named reason when it does not).
    static var isAvailable: Bool { MetalEngine.shared != nil }
}

extension Multislice {
    /// Whether `.metal` can run here.
    public static var metalIsAvailable: Bool { MetalMultislice.isAvailable }
}

extension Multislice {
    /// The default thread cap of a CPU batch (`ExecutionOptions.maximumCPUThreads`, 8): each thread runs whole positions
    /// with its own FFT plan; every exit wave is computed exactly as one sequential run computes it (bit for bit).
    public static let maximumCPUThreads = ExecutionOptions.standard.maximumCPUThreads

    /// The exit waves of `incidents` on at most `threads` threads, in rounds of one wave per thread: the calling task's
    /// cancellation is checked before each round and by every worker after each slice (`CancellationError`, #127); after
    /// a round, `round(first index, exit waves)` gets the round's waves in order. Each wave is computed exactly as one
    /// sequential run computes it. `workspace` holds the FFT plans and propagators the rounds reuse; pass one to reuse it
    /// across calls (chunks). `afterSlice(j)` runs on the worker after slice j, before its cancellation check.
    package static func forEachCPURound(_ incidents: [Waves], potential: ProjectedPotential, transmissions: [ComplexField],
                                        last: Int, threads: Int, workspace: CPUWorkspace? = nil,
                                        afterSlice: (Int) throws -> Void = { _ in },
                                        round: (Int, [Waves]) throws -> Void) throws {
        let count = incidents.count
        let workers = max(1, min(threads, ProcessInfo.processInfo.activeProcessorCount, count))
        let workspace = workspace ?? CPUWorkspace(grid: potential.grid)
        try workspace.reservePlans(workers)
        let thicknesses = Set(potential.slices.map(\.thickness))
        var first = 0
        while first < count {
            try Task.checkCancellation()
            let size = min(workers, count - first), base = first
            let propagators = (0..<size).map { workspace.propagators(for: incidents[base + $0], thicknesses: thicknesses) }
            let results = Mutex<[Int: Waves]>([:])
            let failure = Mutex<(any Error)?>(nil)
            // The workers are GCD threads, where `Task.isCancelled` is always false: they read the calling task's state
            // through its unsafe reference, whose `isCancelled` any thread may read while this closure runs.
            withUnsafeCurrentTask { task in
                DispatchQueue.concurrentPerform(iterations: size) { k in
                    do {
                        // Plan k is this worker's alone during the round.
                        let w = try runCPU(incidents[base + k], potential: potential, transmissions: transmissions,
                                           planes: [last], fft: workspace.plan(k), propagators: propagators[k],
                                           afterSlice: { (j: Int) throws -> Void in
                                               try afterSlice(j)
                                               if task?.isCancelled == true { throw CancellationError() }
                                           })[0]
                        results.withLock { $0[k] = w }
                    } catch {
                        failure.withLock { if $0 == nil { $0 = error } }
                    }
                }
            }
            if let error = failure.withLock({ $0 }) { throw error }
            let waves = results.withLock { r in (0..<size).map { r[$0]! } }
            first += size
            try round(base, waves)
        }
    }

    /// The exit waves (after the last slice) of a batch of incident waves through one potential: on `.cpu` on at most
    /// `options.maximumCPUThreads` threads from the shared transmission functions, on `.metal` in batches whose waves fit
    /// in `options.metalBatchBytes`, the transmissions built on the GPU.
    ///
    /// - Throws: `SimulatoError.gridMismatch` for a wave on another grid, `mixedEnergies` for waves of different energies,
    ///   `invalidTilt` for different tilts on `.metal`.
    ///
    /// Cancellation is checked after every slice of every wave on `.cpu` and every chunk of slices on `.metal`
    /// (`CancellationError`); `progress` counts the waves done, at most once per `options.progressInterval`.
    public static func runBatch(_ incidents: [Waves], through potential: ProjectedPotential,
                                backend: Backend = .cpu, options: ExecutionOptions = .standard,
                                progress: SimulationProgressHandler? = nil) throws -> [Waves] {
        guard let first = incidents.first else { return [] }
        for w in incidents where !w.grid.matches(potential.grid) { throw SimulatoError.gridMismatch }
        try checkSharedEnergy(incidents)
        var reporter = SimulationProgressReporter(progress, interval: options.progressInterval)
        reporter.report(batchPreparing(incidents.count))
        try Task.checkCancellation()
        let tilt = backend == .metal ? try sharedTilt(incidents) : first.tilt
        let prepared = try prepareBatch(potential, beam: first.beam, tilt: tilt, backend: backend, options: options)
        return try runPrepared(incidents, prepared, options, &reporter)
    }

    /// The same for a `Potential`: on `.cpu` built first (`build()`), on `.metal` built slice by slice on the GPU.
    public static func runBatch(_ incidents: [Waves], through potential: Potential,
                                backend: Backend = .cpu, options: ExecutionOptions = .standard,
                                progress: SimulationProgressHandler? = nil) throws -> [Waves] {
        guard let first = incidents.first else { return [] }
        for w in incidents where !w.grid.matches(potential.grid) { throw SimulatoError.gridMismatch }
        try checkSharedEnergy(incidents)
        var reporter = SimulationProgressReporter(progress, interval: options.progressInterval)
        reporter.report(batchPreparing(incidents.count))
        try Task.checkCancellation()
        let tilt = backend == .metal ? try sharedTilt(incidents) : first.tilt
        let prepared = try prepareBatch(potential, beam: first.beam, tilt: tilt, backend: backend, options: options)
        return try runPrepared(incidents, prepared, options, &reporter)
    }

    /// `potential` made ready once for many batches of `beam`'s energy (and, on `.metal`, of `tilt`): on `.cpu` the
    /// transmission functions and a workspace of FFT plans and propagators, on `.metal` the slices on the GPU. A scan
    /// that runs its positions in chunks prepares once and runs every chunk with `runBatch(_:prepared:options:progress:)`.
    /// On `.metal`, `options` sets the transmission cache and its renormalisation: run with the same options.
    /// On `.cpu` a `workspace` made for the same grid (one per scan, kept across its configurations) gives its FFT
    /// plans and propagators to the batch; without one, or for another grid, a new one is made.
    package static func prepareBatch(_ potential: ProjectedPotential, beam: ElectronBeam, tilt: SIMD2<Double>,
                                     backend: Backend, options: ExecutionOptions = .standard,
                                     workspace: CPUWorkspace? = nil) throws -> PreparedBatch {
        switch backend {
        case .cpu:
            let transmissions = try transmissionFunctions(potential, beam: beam)
            let workspace = workspace?.reused(for: potential.grid) ?? CPUWorkspace(grid: potential.grid)
            return PreparedBatch(storage: .cpu(potential, transmissions, workspace),
                                 grid: potential.grid, energy: beam.energy, tilt: tilt)
        case .metal:
            let engine = try MetalEngine.require()
            let prepared = try MetalMultislice.prepare(potential, beam: beam, tilt: tilt, options: options, engine: engine)
            return PreparedBatch(storage: .metal(prepared), grid: potential.grid, energy: beam.energy, tilt: tilt)
        }
    }

    /// The same for a `Potential`: on `.cpu` built here (`build()`), on `.metal` built slice by slice on the GPU.
    package static func prepareBatch(_ potential: Potential, beam: ElectronBeam, tilt: SIMD2<Double>,
                                     backend: Backend, options: ExecutionOptions = .standard,
                                     workspace: CPUWorkspace? = nil) throws -> PreparedBatch {
        switch backend {
        case .cpu:
            let built = try potential.build()
            try Task.checkCancellation()
            return try prepareBatch(built, beam: beam, tilt: tilt, backend: .cpu, workspace: workspace)
        case .metal:
            let engine = try MetalEngine.require()
            let prepared = try MetalMultislice.prepare(potential, beam: beam, tilt: tilt, options: options, engine: engine)
            return PreparedBatch(storage: .metal(prepared), grid: potential.grid, energy: beam.energy, tilt: tilt)
        }
    }

    /// The exit waves of a batch through a prepared potential, as `runBatch(_:through:backend:options:progress:)`
    /// computes them.
    /// - Throws: `SimulatoError.gridMismatch`, `mixedEnergies` for a wave of another energy than the preparation's,
    ///   `invalidTilt` on `.metal` for a wave of another tilt.
    package static func runBatch(_ incidents: [Waves], prepared: PreparedBatch, options: ExecutionOptions = .standard,
                                 progress: SimulationProgressHandler? = nil) throws -> [Waves] {
        guard let first = incidents.first else { return [] }
        for w in incidents where !w.grid.matches(prepared.grid) { throw SimulatoError.gridMismatch }
        try checkSharedEnergy(incidents)
        guard first.beam.energy == prepared.energy else {
            throw SimulatoError.mixedEnergies(first: prepared.energy, other: first.beam.energy)
        }
        if case .metal = prepared.storage, try sharedTilt(incidents) != prepared.tilt {
            throw SimulatoError.invalidTilt("the waves of a .metal batch must have the tilt its potential was prepared for")
        }
        var reporter = SimulationProgressReporter(progress, interval: options.progressInterval)
        try Task.checkCancellation()
        return try runPrepared(incidents, prepared, options, &reporter)
    }

    static func runPrepared(_ incidents: [Waves], _ prepared: PreparedBatch, _ options: ExecutionOptions,
                            _ reporter: inout SimulationProgressReporter) throws -> [Waves] {
        switch prepared.storage {
        case .cpu(let potential, let transmissions, let workspace):
            return try cpuBatch(incidents, potential, transmissions, workspace, options, &reporter)
        case .metal(let metal):
            return try metalBatch(incidents, metal, options, &reporter)
        }
    }

    static func batchPreparing(_ count: Int) -> SimulationProgress {
        SimulationProgress(stage: .preparing, completed: 0, total: count, unit: .positions, fraction: nil)
    }

    static func cpuBatch(_ incidents: [Waves], _ potential: ProjectedPotential, _ transmissions: [ComplexField],
                         _ workspace: CPUWorkspace, _ options: ExecutionOptions,
                         _ reporter: inout SimulationProgressReporter) throws -> [Waves] {
        let count = incidents.count
        var out: [Waves] = []
        out.reserveCapacity(count)
        try forEachCPURound(incidents, potential: potential, transmissions: transmissions,
                            last: potential.slices.count - 1, threads: options.maximumCPUThreads,
                            workspace: workspace) { _, waves in
            out += waves
            reporter.report(SimulationProgress(stage: .propagating, completed: out.count, total: count, unit: .positions,
                                               fraction: Double(out.count) / Double(count)))
        }
        return out
    }

    /// A batch shares its transmission functions and propagators, so its waves must share their energy (both backends).
    static func checkSharedEnergy(_ incidents: [Waves]) throws {
        let first = incidents[0].beam.energy
        if let other = incidents.first(where: { $0.beam.energy != first }) {
            throw SimulatoError.mixedEnergies(first: first, other: other.beam.energy)
        }
    }

    /// A Metal batch shares its propagators, so its waves must share their tilt.
    static func sharedTilt(_ incidents: [Waves]) throws -> SIMD2<Double> {
        let first = incidents[0].tilt
        guard incidents.allSatisfy({ $0.tilt == first }) else {
            throw SimulatoError.invalidTilt("the waves of one .metal batch must share one tilt; run each tilt separately")
        }
        return first
    }

    static func metalBatch(_ incidents: [Waves], _ prepared: MetalPreparedPotential, _ options: ExecutionOptions,
                           _ reporter: inout SimulationProgressReporter) throws -> [Waves] {
        let count = incidents.count, slices = max(1, prepared.sliceCount)
        let fields = try MetalMultislice.runBatch(incidents.map(\.field), prepared: prepared,
                                                  options: options) { start, done, used in
            let fraction = (Double(start) + Double(used) * Double(min(done, slices)) / Double(slices)) / Double(count)
            reporter.report(SimulationProgress(stage: .propagating, completed: done >= slices ? start + used : start,
                                               total: count, unit: .positions, fraction: fraction))
        }
        return zip(incidents, fields).map {
            Waves.unchecked(field: $0.1, grid: prepared.grid, beam: $0.0.beam, normalization: $0.0.normalization, tilt: $0.0.tilt)
        }
    }
}

extension Multislice {
    /// For the tests: V of every slice of `potential` as the GPU builds it.
    package static func gpuPotentialSlices(_ potential: Potential) throws -> [[Double]] {
        try MetalMultislice.gpuPotentialSlices(potential)
    }

    /// For the tests: the probes Ψ₀ shifted by the ramps, built on the GPU.
    package static func gpuProbes(spectrum: ComplexField, ramps: [(x: ComplexField, y: ComplexField)],
                                  grid: Grid) throws -> [ComplexField] {
        try MetalMultislice.gpuProbes(spectrum: spectrum, ramps: ramps, grid: grid)
    }

    /// For the tests: `.metal` with the per-step renormalisation on or off.
    package static func runMetal(_ incident: Waves, through potential: Potential, exitPlanes: ExitPlanes = .last,
                                 renormalize: Bool) throws -> [Waves] {
        let engine = try MetalEngine.require()
        let options = ExecutionOptions(metalRenormalization: renormalize)
        let prepared = try MetalMultislice.prepare(potential, beam: incident.beam, tilt: incident.tilt, options: options,
                                                   engine: engine)
        return try MetalMultislice.run(incident, prepared: prepared,
                                       planes: exitPlanes.planes(sliceCount: potential.sliceCount),
                                       options: options)
    }

    package static func runMetal(_ incident: Waves, through potential: ProjectedPotential, exitPlanes: ExitPlanes = .last,
                                 renormalize: Bool) throws -> [Waves] {
        let engine = try MetalEngine.require()
        let options = ExecutionOptions(metalRenormalization: renormalize)
        let prepared = try MetalMultislice.prepare(potential, beam: incident.beam, tilt: incident.tilt, options: options,
                                                   engine: engine)
        return try MetalMultislice.run(incident, prepared: prepared,
                                       planes: exitPlanes.planes(sliceCount: potential.slices.count),
                                       options: options)
    }
}

extension Multislice {
    /// The Metal device's largest allocated size [bytes] seen at a commit since the last reset (for the memory report);
    /// `reset` starts a new measurement at the current size.
    package static func metalPeakAllocatedBytes(reset: Bool = false) -> Int {
        guard let engine = MetalEngine.shared else { return 0 }
        if reset { engine.peak.withLock { $0 = engine.device.currentAllocatedSize } }
        return engine.peak.withLock { $0 }
    }
}

extension Multislice {
    /// For the tests: how many compiled Metal executables the process holds (bounded by
    /// `metalExecutableCacheLimit`).
    package static var metalCachedExecutables: Int { MetalEngine.shared?.cachedExecutables ?? 0 }

    /// For the tests: `body` with no other Metal run in between (so a count read in it is the body's own).
    package static func withMetalExclusive<T>(_ body: () throws -> T) rethrows -> T {
        guard let engine = MetalEngine.shared else { return try body() }
        engine.runLock.lock()
        defer { engine.runLock.unlock() }
        return try body()
    }
}

extension Multislice {
    /// For the tests: the Metal batches of a run of `count` waves on `grid`, as (first index, waves used, batch compiled).
    package static func metalBatchPlan(count: Int, grid: Grid, options: ExecutionOptions = .standard) -> [[Int]] {
        MetalMultislice.batchPlan(count: count, grid: grid, options: options).map { [$0.start, $0.used, $0.compiled] }
    }
}

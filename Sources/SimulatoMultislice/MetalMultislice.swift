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

    /// The exit waves of `incidents` on at most `threads` threads, in rounds of one wave per thread: before each round,
    /// on the calling thread, the task's cancellation is checked (`CancellationError`); after it, `round(first index,
    /// exit waves)` gets the round's waves in order. Each wave is computed exactly as one sequential run computes it.
    package static func forEachCPURound(_ incidents: [Waves], potential: ProjectedPotential, transmissions: [ComplexField],
                                        last: Int, threads: Int, round: (Int, [Waves]) throws -> Void) throws {
        let count = incidents.count
        let workers = max(1, min(threads, ProcessInfo.processInfo.activeProcessorCount, count))
        var first = 0
        while first < count {
            try Task.checkCancellation()
            let size = min(workers, count - first), base = first
            let results = Mutex<[Int: Waves]>([:])
            let failure = Mutex<(any Error)?>(nil)
            DispatchQueue.concurrentPerform(iterations: size) { k in
                do {
                    let w = try runCPU(incidents[base + k], potential: potential, transmissions: transmissions,
                                       planes: [last])[0]
                    results.withLock { $0[k] = w }
                } catch {
                    failure.withLock { if $0 == nil { $0 = error } }
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
    /// Cancellation is checked after every round of one wave per thread on `.cpu` and every chunk of slices on `.metal`
    /// (`CancellationError`); `progress` counts the waves done, at most once per `options.progressInterval`.
    public static func runBatch(_ incidents: [Waves], through potential: ProjectedPotential,
                                backend: Backend = .cpu, options: ExecutionOptions = .standard,
                                progress: SimulationProgressHandler? = nil) throws -> [Waves] {
        guard let first = incidents.first else { return [] }
        for w in incidents where !w.grid.matches(potential.grid) { throw SimulatoError.gridMismatch }
        try checkSharedEnergy(incidents)
        var reporter = ProgressReporter(progress, interval: options.progressInterval)
        reporter.report(batchPreparing(incidents.count))
        try Task.checkCancellation()
        switch backend {
        case .cpu:
            return try cpuBatch(incidents, potential, options, &reporter)
        case .metal:
            let engine = try MetalEngine.require()
            let prepared = try MetalMultislice.prepare(potential, beam: first.beam, tilt: try sharedTilt(incidents),
                                                       engine: engine)
            return try metalBatch(incidents, prepared, options, &reporter)
        }
    }

    /// The same for a `Potential`: on `.cpu` built first (`build()`), on `.metal` built slice by slice on the GPU.
    public static func runBatch(_ incidents: [Waves], through potential: Potential,
                                backend: Backend = .cpu, options: ExecutionOptions = .standard,
                                progress: SimulationProgressHandler? = nil) throws -> [Waves] {
        guard let first = incidents.first else { return [] }
        for w in incidents where !w.grid.matches(potential.grid) { throw SimulatoError.gridMismatch }
        try checkSharedEnergy(incidents)
        var reporter = ProgressReporter(progress, interval: options.progressInterval)
        reporter.report(batchPreparing(incidents.count))
        try Task.checkCancellation()
        switch backend {
        case .cpu:
            let built = try potential.build()
            try Task.checkCancellation()
            return try cpuBatch(incidents, built, options, &reporter)
        case .metal:
            let engine = try MetalEngine.require()
            let prepared = try MetalMultislice.prepare(potential, beam: first.beam, tilt: try sharedTilt(incidents),
                                                       engine: engine)
            return try metalBatch(incidents, prepared, options, &reporter)
        }
    }

    static func batchPreparing(_ count: Int) -> SimulationProgress {
        SimulationProgress(stage: .preparing, completed: 0, total: count, unit: .positions, fraction: nil)
    }

    static func cpuBatch(_ incidents: [Waves], _ potential: ProjectedPotential, _ options: ExecutionOptions,
                         _ reporter: inout ProgressReporter) throws -> [Waves] {
        let transmissions = try transmissionFunctions(potential, beam: incidents[0].beam)
        let count = incidents.count
        var out: [Waves] = []
        out.reserveCapacity(count)
        try forEachCPURound(incidents, potential: potential, transmissions: transmissions,
                            last: potential.slices.count - 1, threads: options.maximumCPUThreads) { _, waves in
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
                           _ reporter: inout ProgressReporter) throws -> [Waves] {
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
        let prepared = try MetalMultislice.prepare(potential, beam: incident.beam, tilt: incident.tilt, engine: engine)
        return try MetalMultislice.run(incident, prepared: prepared,
                                       planes: exitPlanes.planes(sliceCount: potential.sliceCount),
                                       options: ExecutionOptions(metalRenormalization: renormalize))
    }

    package static func runMetal(_ incident: Waves, through potential: ProjectedPotential, exitPlanes: ExitPlanes = .last,
                                 renormalize: Bool) throws -> [Waves] {
        let engine = try MetalEngine.require()
        let prepared = try MetalMultislice.prepare(potential, beam: incident.beam, tilt: incident.tilt, engine: engine)
        return try MetalMultislice.run(incident, prepared: prepared,
                                       planes: exitPlanes.planes(sliceCount: potential.slices.count),
                                       options: ExecutionOptions(metalRenormalization: renormalize))
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

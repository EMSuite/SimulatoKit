//
//  STEMRun.swift
//  Role: the STEM loop as a value that can stop and continue: configurations outer, positions inner, every finished
//        position added to its detector sums in configuration order, so a run stopped (cancelled) and continued gives
//        the numbers of an uninterrupted run (bit for bit on .cpu, within the registered tolerance on .metal).
//

import SimulatoCore
import SimulatoMultislice
import SimulatoProbe

/// A STEM scan that can stop and continue.
///
/// ``run(progress:)`` scans the configurations in order and, within one, the positions in order. When its task is
/// cancelled it throws `CancellationError` after the current round of positions (one position per CPU thread) or chunk of
/// slices (Metal) and keeps every position finished so far; calling ``run(progress:)`` again continues from there. The
/// configurations' seeds are fixed by their index (`FrozenPhonons.configurationSeed`), so a continued run equals an
/// uninterrupted one. The run is a value: keep the one that was stopped, and continue that one.
public struct STEMRun: Sendable {
    enum Source: Sendable {
        case built([ProjectedPotential])
        case lazy([Potential])
    }

    public let probe: Probe
    let source: Source
    let positions: [SIMD2<Double>]
    /// The scan's shape: one axis for a line scan, two for a grid.
    public let scanShape: [Int]
    let detectors: [any Detector]
    let shapes: [[Int]]
    let counts: [Int]
    public let backend: Backend
    public let options: ExecutionOptions
    /// Per detector, the values of every position summed over the configurations it has finished.
    var sums: [[Double]]
    /// The configurations finished at every position.
    public private(set) var completedConfigurations = 0
    /// The positions finished in the configuration under way (they hold one configuration more than the others).
    public private(set) var completedPositions = 0

    /// Through built potentials, one per frozen-phonon configuration (one for a static structure).
    /// - Throws: `SimulatoError.invalidScan` for no potential or a scan whose shape does not hold its positions,
    ///   `gridMismatch` for a potential on another grid than the probe, and the detectors' refusals of the grid.
    public init(probe: Probe, potentials: [ProjectedPotential], scan: some ScanPattern, detectors: [any Detector],
                backend: Backend = .cpu, options: ExecutionOptions = .standard) throws {
        guard !potentials.isEmpty else { throw SimulatoError.invalidScan("no potential") }
        try STEM.checkShape(scan)
        for p in potentials where !p.grid.matches(probe.grid) { throw SimulatoError.gridMismatch }
        try self.init(probe: probe, source: .built(potentials), scan: scan, detectors: detectors, backend: backend,
                      options: options)
    }

    /// Through unbuilt potentials: on `.cpu` each is built when its configuration starts, on `.metal` each slice is
    /// built on the GPU.
    public init(probe: Probe, potentials: [Potential], scan: some ScanPattern, detectors: [any Detector],
                backend: Backend = .cpu, options: ExecutionOptions = .standard) throws {
        guard !potentials.isEmpty else { throw SimulatoError.invalidScan("no potential") }
        try STEM.checkShape(scan)
        for p in potentials where !p.grid.matches(probe.grid) { throw SimulatoError.gridMismatch }
        try self.init(probe: probe, source: .lazy(potentials), scan: scan, detectors: detectors, backend: backend,
                      options: options)
    }

    init(probe: Probe, source: Source, scan: some ScanPattern, detectors: [any Detector], backend: Backend,
         options: ExecutionOptions) throws {
        self.probe = probe
        self.source = source
        positions = scan.positions
        scanShape = scan.shape
        self.detectors = detectors
        let shapes = try detectors.map { try $0.shape(for: probe.grid, beam: probe.beam) }
        self.shapes = shapes
        counts = shapes.map { $0.reduce(1, *) }
        sums = counts.map { [Double](repeating: 0, count: $0 * scan.positions.count) }
        self.backend = backend
        self.options = options
    }

    /// The number of configurations (potentials).
    public var configurationCount: Int {
        switch source {
        case .built(let p): return p.count
        case .lazy(let p): return p.count
        }
    }

    /// The number of scan positions.
    public var positionCount: Int { positions.count }

    /// Whether every configuration is finished.
    public var isComplete: Bool { completedConfigurations == configurationCount }

    /// Each detector's measurement so far: at every position the mean over the configurations finished there (the
    /// positions before ``completedPositions`` include the configuration under way); zero where none is. Complete, it is
    /// the mean over every configuration, as `STEM.scan` returns it.
    public func measurements() -> [ScanMeasurement] {
        (0..<detectors.count).map { d in
            let c = counts[d]
            var values = sums[d]
            for q in 0..<positions.count {
                let n = completedConfigurations + (q < completedPositions ? 1 : 0)
                for i in 0..<c {
                    values[q * c + i] = n > 0 ? values[q * c + i] / Double(n) : 0
                }
            }
            return ScanMeasurement.unchecked(scanShape: scanShape, valueShape: shapes[d], values: values)
        }
    }

    /// Scans what is left and returns ``measurements()``. `progress` gets the stage, the configuration and the positions
    /// done in it, at most once per `options.progressInterval`, never after a cancel.
    /// - Throws: `CancellationError` when the task is cancelled (what was finished is kept: call again to continue); the
    ///   simulation's errors (after one of those the run's sums may hold part of a round: start a new run).
    public mutating func run(progress: SimulationProgressHandler? = nil) throws -> [ScanMeasurement] {
        var reporter = ProgressReporter(progress, interval: options.progressInterval)
        while completedConfigurations < configurationCount {
            reporter.report(snapshot(.preparing, slicesDone: 0, slices: 1, fractionKnown: false))
            try Task.checkCancellation()
            switch backend {
            case .cpu: try cpuConfiguration(&reporter)
            case .metal: try metalConfiguration(&reporter)
            }
            completedConfigurations += 1
            completedPositions = 0
        }
        return measurements()
    }

    /// The progress now, with `slicesDone` of `slices` of the next positions' batch propagated.
    func snapshot(_ stage: SimulationProgress.Stage, slicesDone: Int = 0, slices: Int = 1, batch: Int = 0,
                fractionKnown: Bool = true) -> SimulationProgress {
        let k = configurationCount, p = positions.count
        let partial = Double(batch) * Double(min(slicesDone, slices)) / Double(max(1, slices))
        let done = Double(completedConfigurations) * Double(p) + Double(completedPositions) + partial
        let whole = Double(k) * Double(p)
        return SimulationProgress(stage: stage, completed: completedPositions, total: p, unit: .positions,
                                  fraction: fractionKnown ? (whole > 0 ? done / whole : 1) : nil,
                                  configuration: k > 1 ? completedConfigurations : nil,
                                  configurationCount: k > 1 ? k : nil)
    }

    mutating func cpuConfiguration(_ reporter: inout ProgressReporter) throws {
        let potential: ProjectedPotential
        switch source {
        case .built(let p): potential = p[completedConfigurations]
        case .lazy(let p): potential = try p[completedConfigurations].build()
        }
        try Task.checkCancellation()
        let transmissions = try Multislice.transmissionFunctions(potential, beam: probe.beam)
        // Positions in chunks of `options.scanChunkBytes`, so the waves of one chunk only are held (13 MB per position
        // at 480²).
        let chunk = max(1, options.scanChunkBytes / (probe.grid.count * 16 * 2))
        let probe = probe, positions = positions, detectors = detectors, counts = counts
        var sums: [[Double]] = []
        swap(&sums, &self.sums)
        defer { swap(&sums, &self.sums) }
        while completedPositions < positions.count {
            let first = completedPositions
            let incidents = try (first..<min(first + chunk, positions.count)).map { try probe.build(at: positions[$0]) }
            try Multislice.forEachCPURound(incidents, potential: potential, transmissions: transmissions,
                                           last: potential.slices.count - 1,
                                           threads: options.maximumCPUThreads) { base, waves in
                for (offset, exit) in waves.enumerated() {
                    try STEM.accumulate(DiffractionIntensity(exit), at: first + base + offset, detectors, counts, &sums)
                }
                completedPositions = first + base + waves.count
                reporter.report(snapshot(.scanning))
            }
        }
    }

    mutating func metalConfiguration(_ reporter: inout ProgressReporter) throws {
        let start = completedPositions
        let remaining = positions.count - start
        guard remaining > 0 else { return }
        let probe = probe, positions = positions, detectors = detectors, counts = counts, options = options
        let spectrum = probe.centredSpectrum()
        var sums: [[Double]] = []
        swap(&sums, &self.sums)
        defer { swap(&sums, &self.sums) }
        let slices: Int
        switch source {
        case .built(let p): slices = p[completedConfigurations].slices.count
        case .lazy(let p): slices = p[completedConfigurations].sliceCount
        }
        switch source {
        case .built(let p):
            try Multislice.metalScanIntensities(
                spectrum: spectrum, positions: remaining, ramps: { try probe.shiftRamps(at: positions[start + $0]) },
                through: p[completedConfigurations], beam: probe.beam, tilt: probe.tilt, options: options,
                afterChunk: { _, done, used in
                    reporter.report(snapshot(.scanning, slicesDone: done, slices: slices, batch: done < slices ? used : 0))
                },
                deliver: { q, values in
                    let intensity = DiffractionIntensity.unchecked(grid: probe.grid, beam: probe.beam, values: values)
                    try STEM.accumulate(intensity, at: start + q, detectors, counts, &sums)
                    completedPositions = start + q + 1
                })
        case .lazy(let p):
            try Multislice.metalScanIntensities(
                spectrum: spectrum, positions: remaining, ramps: { try probe.shiftRamps(at: positions[start + $0]) },
                through: p[completedConfigurations], beam: probe.beam, tilt: probe.tilt, options: options,
                afterChunk: { _, done, used in
                    reporter.report(snapshot(.scanning, slicesDone: done, slices: slices, batch: done < slices ? used : 0))
                },
                deliver: { q, values in
                    let intensity = DiffractionIntensity.unchecked(grid: probe.grid, beam: probe.beam, values: values)
                    try STEM.accumulate(intensity, at: start + q, detectors, counts, &sums)
                    completedPositions = start + q + 1
                })
        }
    }
}

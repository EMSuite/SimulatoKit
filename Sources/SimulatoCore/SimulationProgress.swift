//
//  SimulationProgress.swift
//  Role: what a long run reports while it computes (the stage, a count with its unit, an overall fraction), the handler
//        that receives it, and the reporter every run loop calls: at most one call per interval, none after a cancel.
//

/// Where a run is and how far it has come.
public struct SimulationProgress: Sendable, Hashable {
    /// The part of the run under way.
    public enum Stage: Sendable, Hashable {
        /// Building the slab from the crystal.
        case buildingSlab
        /// Building the potential, the transmission functions or the GPU program: nothing to count yet.
        case preparing
        /// Propagating waves slice by slice.
        case propagating
        /// Scanning a probe over positions.
        case scanning
    }

    /// What ``completed`` and ``total`` count.
    public enum Unit: Sendable, Hashable {
        case slices
        case positions
        case configurations
    }

    public let stage: Stage
    /// How many units of this stage are done.
    public let completed: Int
    /// How many units this stage has.
    public let total: Int
    public let unit: Unit
    /// The fraction of the whole run done, in [0, 1]; nil while the stage has nothing to count (compiling the GPU program).
    public let fraction: Double?
    /// The frozen-phonon configuration under way, zero-based, and how many the run has (nil for a run of one potential).
    public let configuration: Int?
    public let configurationCount: Int?

    public init(stage: Stage, completed: Int, total: Int, unit: Unit, fraction: Double?, configuration: Int? = nil,
                configurationCount: Int? = nil) {
        self.stage = stage
        self.completed = completed
        self.total = total
        self.unit = unit
        self.fraction = fraction.map { min(1, max(0, $0)) }
        self.configuration = configuration
        self.configurationCount = configurationCount
    }
}

/// Receives a run's progress. It is called on the thread that runs the computation (never the main actor unless the
/// run was started there), between chunks of work, with no lock of the kit held.
public typealias SimulationProgressHandler = @Sendable (SimulationProgress) -> Void

/// Calls a progress handler at most once per `interval` (the first call always), and never once the current task is
/// cancelled.
package struct ProgressReporter {
    let handler: SimulationProgressHandler?
    let interval: Duration
    let now: () -> ContinuousClock.Instant
    var last: ContinuousClock.Instant?

    package init(_ handler: SimulationProgressHandler?, interval: Duration,
                 now: @escaping () -> ContinuousClock.Instant = { ContinuousClock.now }) {
        self.handler = handler
        self.interval = interval
        self.now = now
    }

    /// Whether a handler is attached.
    package var isActive: Bool { handler != nil }

    package mutating func report(_ progress: @autoclosure () -> SimulationProgress) {
        guard let handler, !Task.isCancelled else { return }
        let t = now()
        if let last, t - last < interval { return }
        last = t
        handler(progress())
    }
}

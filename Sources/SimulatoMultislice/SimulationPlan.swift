//
//  SimulationPlan.swift
//  Role: what a run will need before it runs (#91): the sampling from the largest angle the result needs, the grid
//        (abTEM's for that sampling, rounded up to FFT-friendly lengths), the slices, the propagation steps and FFTs,
//        the GPU memory by the design's budget table, and a time from a rate the caller measured. A plan over the
//        memory budget, over the device's largest buffer or over 4096² is refused with the sampling that would fit.
//

import Foundation
import Metal
import SimulatoCore

/// The limits a plan is held to.
public struct SimulationLimits: Sendable, Hashable {
    /// The most memory one run may take [bytes].
    public var memoryBudget: Int
    /// The largest single buffer the device allows [bytes].
    public var largestBuffer: Int
    /// The largest grid length along either axis.
    public var largestGridLength: Int

    public init(memoryBudget: Int, largestBuffer: Int, largestGridLength: Int = 4096) {
        self.memoryBudget = memoryBudget
        self.largestBuffer = largestBuffer
        self.largestGridLength = largestGridLength
    }

    /// This Mac: the budget is the smaller of half of Metal's recommended working set and a quarter of the physical
    /// memory, the largest buffer is the device's `maxBufferLength` (without Metal, both are a quarter of the memory).
    public static var thisMac: SimulationLimits {
        let quarter = Int(min(ProcessInfo.processInfo.physicalMemory / 4, UInt64(Int.max)))
        guard let device = MetalEngine.shared?.device else {
            return SimulationLimits(memoryBudget: quarter, largestBuffer: quarter)
        }
        let working = Int(min(device.recommendedMaxWorkingSetSize / 2, UInt64(Int.max)))
        return SimulationLimits(memoryBudget: min(working, quarter), largestBuffer: device.maxBufferLength)
    }
}

/// A plan refused, with the sampling that would fit (nil when none does: the smallest grid is over the limits).
public enum SimulationPlanError: Error, Sendable, Equatable, CustomStringConvertible {
    /// The run needs `required` bytes, more than the `budget`; `samplingThatFits` [Å] would fit.
    case overBudget(required: Int, budget: Int, samplingThatFits: Double?)
    /// One buffer of the run needs `required` bytes, more than the device's `limit`.
    case bufferTooLarge(required: Int, limit: Int, samplingThatFits: Double?)
    /// The grid is `nx` × `ny`, longer than `limit` along an axis.
    case gridTooLarge(nx: Int, ny: Int, limit: Int, samplingThatFits: Double?)
    /// A request value outside its range.
    case invalidRequest(String)

    public var description: String {
        func fits(_ s: Double?) -> String {
            s.map { "; a sampling of \($0) Å would fit" } ?? "; no sampling fits"
        }
        switch self {
        case let .overBudget(r, b, s):
            return "this run needs about \(r) bytes on the GPU, more than the \(b) bytes allowed" + fits(s)
        case let .bufferTooLarge(r, l, s):
            return "one buffer of this run needs \(r) bytes, more than the device's largest (\(l) bytes)" + fits(s)
        case let .gridTooLarge(nx, ny, l, s):
            return "the grid would be \(nx) x \(ny), longer than \(l) along an axis" + fits(s)
        case let .invalidRequest(why):
            return "invalid simulation request: \(why)"
        }
    }
}

/// What a multislice run will need, computed before it runs.
public struct SimulationPlan: Sendable, Equatable {
    /// How the specimen is lit.
    public enum Illumination: Sendable, Hashable {
        /// One plane wave (HRTEM, parallel-beam diffraction).
        case planeWave
        /// A probe at each of `positions` positions (CBED: 1; a scan: its positions), batched on the GPU.
        case probes(positions: Int)
    }

    /// What to plan.
    public struct Request: Sendable, Hashable {
        /// The box's lateral size (x, y) [Å].
        public var extent: SIMD2<Double>
        /// The specimen thickness [Å].
        public var thickness: Double
        /// The requested slice thickness [Å] (abTEM's default 1 Å).
        public var sliceThickness: Double
        /// The beam energy [eV].
        public var energy: Double
        /// The largest scattering angle the result needs [mrad]: the objective aperture for HRTEM, the outer detector
        /// angle for STEM, the pattern's edge for CBED. The antialias cutoff of the planned grid lands at about this
        /// angle (abTEM's `cutoff_angles` counts whole angular pixels, so it may fall short by a pixel or so).
        public var largestAngle: Double
        /// A sampling [Å] to use instead of the rule.
        public var sampling: Double?
        /// The number of elements in the specimen (one scattering kernel each).
        public var elements: Int
        public var illumination: Illumination
        /// Frozen-phonon configurations (1 for a static lattice).
        public var configurations: Int
        /// Exit waves kept per wave (thicknesses of a series; 1 for the last only).
        public var storedExitPlanes: Int
        /// The GPU memory for one batch of waves [bytes] (`ExecutionOptions.metalBatchBytes`).
        public var batchBytes: Int

        public init(extent: SIMD2<Double>, thickness: Double, sliceThickness: Double = 1, energy: Double,
                    largestAngle: Double, sampling: Double? = nil, elements: Int = 1, illumination: Illumination = .planeWave,
                    configurations: Int = 1, storedExitPlanes: Int = 1,
                    batchBytes: Int = ExecutionOptions.standard.metalBatchBytes) {
            self.extent = extent
            self.thickness = thickness
            self.sliceThickness = sliceThickness
            self.energy = energy
            self.largestAngle = largestAngle
            self.sampling = sampling
            self.elements = elements
            self.illumination = illumination
            self.configurations = configurations
            self.storedExitPlanes = storedExitPlanes
            self.batchBytes = batchBytes
        }
    }

    /// GPU memory by item [bytes], N grid points (the design's budget table).
    public struct Memory: Sendable, Hashable {
        /// The working set of one batch of waves: 24 N per wave.
        public let waves: Int
        /// One scattering kernel per element: 4 N each.
        public let kernels: Int
        /// One propagator per distinct slice thickness: 8 N each.
        public let propagators: Int
        /// The exit waves kept: 8 N for each stored plane of each wave in a batch.
        public let storedExitWaves: Int

        /// The sum: the predicted peak.
        public var total: Int {
            var sum = 0
            for v in [waves, kernels, propagators, storedExitWaves] {
                let (t, o) = sum.addingReportingOverflow(v)
                if o { return .max }
                sum = t
            }
            return sum
        }
    }

    public let request: Request
    /// The grid: abTEM's for the sampling, each length rounded up to factors 2, 3 and 5 (both samplings at most the
    /// requested one).
    public let grid: Grid
    /// The sampling the rule (or the request) asked for [Å]; the grid's own sampling is at most this.
    public let requestedSampling: Double
    /// The scattering angles at the antialias cutoff [mrad] (abTEM's `cutoff_angles`).
    public let cutoffAngles: (x: Double, y: Double)
    /// The slice thicknesses [Å].
    public let sliceThicknesses: [Double]
    /// Waves per batch on the GPU.
    public let batchSize: Int
    /// Slice steps over every wave and configuration.
    public let propagationSteps: Int
    /// FFTs: two per slice step.
    public let fftCount: Int
    public let memory: Memory
    /// The largest single buffer: one batch of complex float32 waves, 8 N per wave [bytes].
    public let largestBufferBytes: Int

    /// The number of slices.
    public var sliceCount: Int { sliceThicknesses.count }

    /// The sampling rule [Å]: the smaller of 0.05 Å and λ / (3 β), β the largest angle in rad, because abTEM's
    /// antialiasing keeps two thirds of the band.
    public static func sampling(largestAngle: Double, energy: Double) throws -> Double {
        guard largestAngle.isFinite, largestAngle > 0 else {
            throw SimulationPlanError.invalidRequest("the largest angle must be finite and positive, got \(largestAngle) mrad")
        }
        let lambda = try ElectronBeam(energy: energy).wavelength
        return min(0.05, lambda / (3 * largestAngle * 1e-3))
    }

    /// The smallest length at least `n` whose only prime factors are 2, 3 and 5.
    public static func fftFriendlyLength(atLeast n: Int) -> Int {
        var m = max(1, n)
        while true {
            var k = m
            for p in [2, 3, 5] { while k % p == 0 { k /= p } }
            if k == 1 { return m }
            m += 1
        }
    }

    /// The time the run will take [s] at `stepsPerSecond` slice steps per second, measured by the caller on this grid.
    /// - Throws: `SimulationPlanError.invalidRequest` unless the rate is finite and positive.
    public func estimatedSeconds(stepsPerSecond: Double) throws -> Double {
        guard stepsPerSecond.isFinite, stepsPerSecond > 0 else {
            throw SimulationPlanError.invalidRequest("the rate must be finite and positive, got \(stepsPerSecond) steps per second")
        }
        return Double(propagationSteps) / stepsPerSecond
    }

    public static func == (a: SimulationPlan, b: SimulationPlan) -> Bool {
        a.request == b.request && a.grid == b.grid && a.requestedSampling == b.requestedSampling
            && a.cutoffAngles == b.cutoffAngles && a.sliceThicknesses == b.sliceThicknesses && a.batchSize == b.batchSize
            && a.propagationSteps == b.propagationSteps && a.memory == b.memory
    }

    /// The plan without the limits.
    static func unchecked(_ r: Request) throws -> SimulationPlan {
        guard r.extent.x.isFinite, r.extent.y.isFinite, r.extent.x > 0, r.extent.y > 0 else {
            throw SimulationPlanError.invalidRequest("the extent must be finite and positive, got \(r.extent) Å")
        }
        guard r.thickness.isFinite, r.thickness > 0 else {
            throw SimulationPlanError.invalidRequest("the thickness must be finite and positive, got \(r.thickness) Å")
        }
        guard r.elements >= 1, r.configurations >= 1, r.storedExitPlanes >= 1, r.batchBytes >= 1 else {
            throw SimulationPlanError.invalidRequest("elements, configurations, stored exit planes and batch bytes must be at least 1")
        }
        let waveCount: Int
        switch r.illumination {
        case .planeWave: waveCount = 1
        case .probes(let n):
            guard n >= 1 else { throw SimulationPlanError.invalidRequest("a probe run needs at least one position, got \(n)") }
            waveCount = n
        }
        let beam = try ElectronBeam(energy: r.energy)
        let sampling = try r.sampling ?? Self.sampling(largestAngle: r.largestAngle, energy: r.energy)
        guard sampling.isFinite, sampling > 0 else {
            throw SimulationPlanError.invalidRequest("the sampling must be finite and positive, got \(sampling) Å")
        }
        let grid = try Self.grid(extent: r.extent, sampling: sampling)
        let thicknesses = try Slicing.thicknesses(depth: r.thickness, requested: r.sliceThickness)
        let n = grid.count
        let (perWave, o0) = n.multipliedReportingOverflow(by: 24)
        let batch = o0 ? 1 : max(1, min(waveCount, r.batchBytes / perWave))
        func product(_ values: Int...) -> Int {
            var p = 1
            for v in values {
                let (q, o) = p.multipliedReportingOverflow(by: v)
                if o { return .max }
                p = q
            }
            return p
        }
        let memory = Memory(waves: product(24, n, batch), kernels: product(4, n, r.elements),
                            propagators: product(8, n, Set(thicknesses).count),
                            storedExitWaves: product(8, n, r.storedExitPlanes, min(waveCount, batch)))
        let steps = product(thicknesses.count, waveCount, r.configurations)
        return SimulationPlan(request: r, grid: grid, requestedSampling: sampling,
                              cutoffAngles: Antialias.cutoffAngles(on: grid, wavelength: beam.wavelength),
                              sliceThicknesses: thicknesses, batchSize: batch, propagationSteps: steps,
                              fftCount: product(2, steps), memory: memory, largestBufferBytes: product(8, n, batch))
    }

    /// The longest grid axis a plan builds (longer is refused before any limit is applied).
    static let searchLength = 1 << 16

    /// abTEM's gpts for the sampling (⌈L / Δ⌉, at least 2), each rounded up to the next length with factors 2, 3 and 5.
    /// The axes are rounded on their own: both samplings stay at most Δ, and they may differ by the rounding (up to
    /// about 11 %), because asking them to agree within 1 % can inflate the grid many times over.
    static func grid(extent: SIMD2<Double>, sampling: Double) throws -> Grid {
        let limit = Double(searchLength)
        let rx = max(2, (extent.x / sampling).rounded(.up)), ry = max(2, (extent.y / sampling).rounded(.up))
        guard rx <= limit, ry <= limit else {
            throw SimulationPlanError.gridTooLarge(nx: rx <= limit ? Int(rx) : .max, ny: ry <= limit ? Int(ry) : .max,
                                                   limit: searchLength, samplingThatFits: nil)
        }
        return try Grid(nx: fftFriendlyLength(atLeast: Int(rx)), ny: fftFriendlyLength(atLeast: Int(ry)),
                        extentX: extent.x, extentY: extent.y)
    }

    /// The first limit this plan breaks, with the sampling that would fit, or nil.
    func violation(of limits: SimulationLimits) -> SimulationPlanError? {
        if max(grid.nx, grid.ny) > limits.largestGridLength {
            return .gridTooLarge(nx: grid.nx, ny: grid.ny, limit: limits.largestGridLength,
                                 samplingThatFits: Self.fittingSampling(request, limits, from: requestedSampling))
        }
        if largestBufferBytes > limits.largestBuffer {
            return .bufferTooLarge(required: largestBufferBytes, limit: limits.largestBuffer,
                                   samplingThatFits: Self.fittingSampling(request, limits, from: requestedSampling))
        }
        if memory.total > limits.memoryBudget {
            return .overBudget(required: memory.total, budget: limits.memoryBudget,
                               samplingThatFits: Self.fittingSampling(request, limits, from: requestedSampling))
        }
        return nil
    }

    /// The excess of a plan over `limits` (at most 1 when it fits).
    func excess(over limits: SimulationLimits) -> Double {
        max(Double(memory.total) / Double(max(1, limits.memoryBudget)),
            Double(largestBufferBytes) / Double(max(1, limits.largestBuffer)),
            Double(max(grid.nx, grid.ny)) / Double(max(1, limits.largestGridLength)))
    }

    /// A sampling at least `start` that plans `request` within `limits`, or nil when none does. Memory scales about as
    /// 1 / Δ², so each try coarsens by the square root of the excess plus 1 % (at most twofold); nil once the grid is
    /// 2 x 2 and still over, or after 200 tries.
    static func fittingSampling(_ request: Request, _ limits: SimulationLimits, from start: Double) -> Double? {
        var s = start
        for _ in 0..<200 {
            var r = request
            r.sampling = s
            let plan: SimulationPlan
            do {
                plan = try unchecked(r)
            } catch SimulationPlanError.gridTooLarge {
                s *= 2
                continue
            } catch {
                return nil
            }
            let e = plan.excess(over: limits)
            if e <= 1 { return s }
            if plan.grid.nx <= 2, plan.grid.ny <= 2 { return nil }
            s *= min(2, max(e.squareRoot(), 1) * 1.01)
        }
        return nil
    }
}

extension SimulationPlan {
    /// Plans `request` and holds it to `limits`.
    /// - Throws: `SimulationPlanError.invalidRequest` for a value outside its range, `gridTooLarge`, `bufferTooLarge`
    ///   or `overBudget` (each with the sampling that would fit), `SimulatoError.invalidEnergy` and the slicing's
    ///   refusals.
    public init(_ request: Request, limits: SimulationLimits = .thisMac) throws {
        let plan: SimulationPlan
        do {
            plan = try Self.unchecked(request)
        } catch SimulationPlanError.gridTooLarge(let nx, let ny, _, _) {
            let start = max(request.extent.x, request.extent.y) / Double(limits.largestGridLength)
            throw SimulationPlanError.gridTooLarge(nx: nx, ny: ny, limit: limits.largestGridLength,
                                                   samplingThatFits: Self.fittingSampling(request, limits, from: start))
        }
        if let error = plan.violation(of: limits) {
            throw error
        }
        self = plan
    }
}

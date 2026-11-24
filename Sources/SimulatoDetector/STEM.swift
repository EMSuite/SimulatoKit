//
//  STEM.swift
//  Role: the STEM driver's entry points and the measurement type: every configuration, every scan position, the probe
//        there through that configuration's potential, each detector on the exit wave, the mean over configurations
//        (abTEM's `ensemble_mean`). The loop itself, which can stop and continue, is `STEMRun`.
//

import Foundation
import SimulatoCore
import SimulatoMultislice
import SimulatoProbe

/// One detector's measurement over a scan: values row-major over (scan x, scan y, the detector's shape). Named
/// `ScanMeasurement`, not abTEM's `Measurement`, so it does not collide with Foundation's `Measurement`.
public struct ScanMeasurement: Sendable {
    public let scanShape: [Int]
    public let valueShape: [Int]
    public let values: [Double]

    /// - Throws: `SimulatoError.invalidShape` unless `scanShape` has one axis (a `LineScan`) or two (a `GridScan`), every
    ///   axis is non-negative, and `values` holds the product of `scanShape` × the product of `valueShape` numbers.
    public init(scanShape: [Int], valueShape: [Int], values: [Double]) throws {
        guard Self.fills(scanShape, valueShape, values.count) else {
            throw SimulatoError.invalidShape("scan shape \(scanShape) (one or two axes) × value shape \(valueShape) for \(values.count) values")
        }
        self.init(storing: scanShape, valueShape, values)
    }

    /// The kit's own construction, from arrays it sized: a mismatch is a bug of the kit and traps.
    package static func unchecked(scanShape: [Int], valueShape: [Int], values: [Double]) -> ScanMeasurement {
        precondition(fills(scanShape, valueShape, values.count), "ScanMeasurement: the values do not fill the shapes")
        return ScanMeasurement(storing: scanShape, valueShape, values)
    }

    /// Whether `count` values fill one or two scan axes × the value shape (false on a negative axis or an overflow).
    static func fills(_ scanShape: [Int], _ valueShape: [Int], _ count: Int) -> Bool {
        guard (1...2).contains(scanShape.count) else { return false }
        var product = 1
        for n in scanShape + valueShape {
            guard n >= 0 else { return false }
            let (r, overflow) = product.multipliedReportingOverflow(by: n)
            if overflow { return false }
            product = r
        }
        return product == count
    }

    private init(storing scanShape: [Int], _ valueShape: [Int], _ values: [Double]) {
        self.scanShape = scanShape
        self.valueShape = valueShape
        self.values = values
    }

    var valueCount: Int { valueShape.reduce(1, *) }

    /// The values at scan position (ix, iy).
    public subscript(ix: Int, iy: Int) -> [Double] {
        precondition(scanShape.count == 2, "ScanMeasurement: (ix, iy) needs a two-axis scan; use the one-index subscript for a line scan")
        let base = (ix * scanShape[1] + iy) * valueCount
        return Array(values[base..<(base + valueCount)])
    }

    /// The image of a one-number detector (annular), sampled as the scan.
    public func image(samplingX: Double, samplingY: Double) -> RealField? {
        guard valueCount == 1, scanShape.count == 2 else { return nil }
        return RealField.unchecked(nx: scanShape[0], ny: scanShape[1], values: values, samplingX: samplingX, samplingY: samplingY)
    }
}

public enum STEM {
    /// Scans `probe` over `scan` through every potential (one per frozen-phonon configuration; one for a static
    /// structure) and returns each detector's measurement averaged over the potentials. On `.metal` the probes, the
    /// transmissions and the diffraction intensities are computed on the GPU.
    ///
    /// The scan stops with `CancellationError` when its task is cancelled (`STEMRun` keeps what was done and continues);
    /// `progress` gets the configuration and the positions done, at most once per `options.progressInterval`.
    public static func scan(probe: Probe, potentials: [ProjectedPotential], scan: some ScanPattern, detectors: [any Detector],
                            backend: Backend = .cpu, options: ExecutionOptions = .standard,
                            progress: SimulationProgressHandler? = nil) throws -> [ScanMeasurement] {
        var run = try STEMRun(probe: probe, potentials: potentials, scan: scan, detectors: detectors, backend: backend,
                              options: options)
        return try run.run(progress: progress)
    }

    /// The same through unbuilt potentials: on `.cpu` each is built (`build()`), on `.metal` each slice is built on the GPU.
    public static func scan(probe: Probe, potentials: [Potential], scan: some ScanPattern, detectors: [any Detector],
                            backend: Backend = .cpu, options: ExecutionOptions = .standard,
                            progress: SimulationProgressHandler? = nil) throws -> [ScanMeasurement] {
        var run = try STEMRun(probe: probe, potentials: potentials, scan: scan, detectors: detectors, backend: backend,
                              options: options)
        return try run.run(progress: progress)
    }

    /// Half of the physical memory [bytes]: the most a scan's detector sums and positions may take.
    static var defaultMemoryLimit: Int { Int(ProcessInfo.processInfo.physicalMemory / 2) }

    /// Refuses, before anything is allocated (the positions included), a scan whose detector sums (8 bytes a value at
    /// every position) and positions (16 bytes each) would take more than `limit` bytes: a pixelated detector over a
    /// large scan holds every pattern (#61).
    /// - Throws: `SimulatoError.invalidScan` naming the positions, the values per position and the limit.
    package static func checkMemory(_ scan: some ScanPattern, detectors: [any Detector], probe: Probe,
                                    limit: Int = defaultMemoryLimit) throws {
        var positions = 1, overflow = false
        for n in scan.shape {
            let r = positions.multipliedReportingOverflow(by: max(n, 0))
            positions = r.partialValue
            overflow = overflow || r.overflow
        }
        var values = 0
        for d in detectors {
            let count = try d.shape(for: probe.grid, beam: probe.beam).reduce(1) { acc, n in
                let r = acc.multipliedReportingOverflow(by: max(n, 0))
                overflow = overflow || r.overflow
                return r.partialValue
            }
            let r = values.addingReportingOverflow(count)
            values = r.partialValue
            overflow = overflow || r.overflow
        }
        let (perPosition, o1) = values.multipliedReportingOverflow(by: 8)
        let (withPosition, o2) = perPosition.addingReportingOverflow(16)
        let (bytes, o3) = positions.multipliedReportingOverflow(by: withPosition)
        guard !(overflow || o1 || o2 || o3), bytes <= limit else {
            let positionText = overflow ? "more than \(Int.max)" : "\(positions)"
            throw SimulatoError.invalidScan("\(positionText) positions with \(values) detector values each need more than the \(limit) bytes allowed (half of the physical memory); scan fewer positions or use a smaller detector")
        }
    }

    /// A custom `ScanPattern` must have one or two axes, none negative, whose product is its number of positions (two negative
    /// axes multiply to a positive count, and the measurement at the end of the scan would trap on them, #128).
    static func checkShape(_ scan: some ScanPattern) throws {
        let shape = scan.shape
        let (product, overflow) = shape.reduce((1, false)) { acc, n in
            let r = acc.0.multipliedReportingOverflow(by: n)
            return (r.partialValue, acc.1 || r.overflow)
        }
        guard (1...2).contains(shape.count), shape.allSatisfy({ $0 >= 0 }), !overflow, product == scan.positions.count else {
            throw SimulatoError.invalidScan("the scan shape \(shape) must have one or two axes whose product is the \(scan.positions.count) positions")
        }
    }

    static func accumulate(_ intensity: DiffractionIntensity, at q: Int, _ detectors: [any Detector], _ counts: [Int],
                           _ sums: inout [[Double]]) throws {
        for (d, detector) in detectors.enumerated() {
            let v = try detector.detect(intensity)
            guard v.count == counts[d] else {
                throw SimulatoError.invalidDetector("detector \(d) returned \(v.count) values; its shape(for:beam:) says \(counts[d])")
            }
            for k in 0..<v.count { sums[d][q * counts[d] + k] += v[k] }
        }
    }
}

//
//  STEM.swift
//  Role: the STEM driver's entry points and the measurement type: every configuration, every scan position, the probe
//        there through that configuration's potential, each detector on the exit wave, the mean over configurations
//        (abTEM's `ensemble_mean`). The loop itself, which can stop and continue, is `STEMRun`.
//

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

    /// A custom `ScanPattern` must have one or two axes whose product is its number of positions.
    static func checkShape(_ scan: some ScanPattern) throws {
        let shape = scan.shape
        let (product, overflow) = shape.reduce((1, false)) { acc, n in
            let r = acc.0.multipliedReportingOverflow(by: n)
            return (r.partialValue, acc.1 || r.overflow)
        }
        guard (1...2).contains(shape.count), !overflow, product == scan.positions.count else {
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

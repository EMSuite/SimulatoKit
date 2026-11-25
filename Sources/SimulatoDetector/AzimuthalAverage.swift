//
//  AzimuthalAverage.swift
//  Role: abTEM 1.0.10's `DiffractionPatterns.azimuthal_average` (measurements.py, `_azimuthal_average`) on a centred
//        diffraction pattern: rings at radii 0, Δ, 2Δ, … below a maximum angle, each the mean of the pixels a step or a
//        Gaussian weight selects around that radius. The angles follow abTEM bit for bit where it matters: the axis
//        coordinates are a float32 `linspace` between the pattern limits [mrad], the radius is float32, the ring test is
//        float64, so a pixel on a ring edge falls on the side abTEM puts it. The sums are float64; abTEM sums a float32
//        pattern in float32, so on float32 data the two differ at about 1e-7 relative.
//

import Foundation
import SimulatoCore

/// How `azimuthalAverage` weights the pixels around each ring (abTEM's `weighting_function`).
public enum AzimuthalWeighting: Sendable, Hashable {
    /// The pixels with |α − ring| < width, each weight 1 (abTEM "step").
    case step
    /// exp(−(α − ring)² / (width² / 2)) over every pixel (abTEM "gaussian").
    case gaussian
}

extension RealField {
    /// The azimuthal average of a centred diffraction pattern (abTEM `DiffractionPatterns.azimuthal_average`): for each
    /// ring radius 0, Δ, 2Δ, … below `maxAngle`, Σ P · w / Σ w with w the weighting around that radius, and 0 where
    /// Σ w is 0. The pattern is centred as `Waves.diffractionPattern` and `PixelatedDetector` return it: sampling in 1/Å,
    /// the zero frequency at index n / 2 of each axis (abTEM `fftshift=True`).
    /// - Parameters:
    ///   - beam: the beam that sets the angles (α = k λ 10³ mrad).
    ///   - maxAngle: the last ring lies below it [mrad]; `nil` is abTEM's default, the largest |lowest angle| of the two
    ///     axes (the edge of the axis that reaches further, not the corner).
    ///   - radialSampling: the ring spacing Δ, in units of the finer angular sampling of the two axes (default 1).
    ///   - weighting: `.step` (default) or `.gaussian`.
    ///   - width: the step half-width or the Gaussian width, in units of the finer angular sampling (default 1).
    /// - Returns: one reciprocal-space profile, sampling Δ / (λ 10³) [1/Å].
    /// - Throws: `invalidDetector` unless the radial sampling, the width and a given maximum angle are finite and
    ///   positive (abTEM returns zeros, NaN or an empty profile), or when the rings would number a million or more.
    public func azimuthalAverage(beam: ElectronBeam, maxAngle: Double? = nil, radialSampling: Double = 1,
                                 weighting: AzimuthalWeighting = .step, width: Double = 1) throws
        -> ReciprocalSpaceLineProfiles {
        guard radialSampling.isFinite, radialSampling > 0 else {
            throw SimulatoError.invalidDetector("the radial sampling must be finite and positive, got \(radialSampling)")
        }
        guard width.isFinite, width > 0 else {
            throw SimulatoError.invalidDetector("the width must be finite and positive, got \(width)")
        }
        if let m = maxAngle, !(m.isFinite && m > 0) {
            throw SimulatoError.invalidDetector("the maximum angle must be finite and positive, got \(m)")
        }
        let wavelength = beam.wavelength
        // abTEM `limits`, then `angular_limits` (k × λ × 10³, in that order), then `angular_sampling`.
        func limits(_ n: Int, _ s: Double) -> (lower: Double, upper: Double) {
            let l = n % 2 == 1 ? (Double(-((n - 1) / 2)) * s, Double((n - 1) / 2) * s)
                               : (Double(-(n / 2)) * s, Double(n / 2 - 1) * s)
            return (l.0 * wavelength * 1e3, l.1 * wavelength * 1e3)
        }
        let lx = limits(nx, samplingX), ly = limits(ny, samplingY)
        let minAngular = Swift.min(samplingX * wavelength * 1e3, samplingY * wavelength * 1e3)
        // abTEM: -min(min(angular_limits)), the lower limit of the lexicographically smaller (lower, upper) pair.
        let maximum = maxAngle ?? -Swift.min(lx.lower, ly.lower)
        let step = radialSampling * minAngular
        let w = width * minAngular

        // NumPy `arange(0, maximum, step)`: ceil(maximum / step) rings at i × step.
        let ringsDouble = (maximum / step).rounded(.up)
        guard ringsDouble < 1e6 else {
            throw SimulatoError.invalidDetector("a maximum angle of \(maximum) mrad at \(step) mrad gives \(ringsDouble) rings")
        }
        let rings = Swift.max(Int(ringsDouble), 0)

        // NumPy `linspace(lower, upper, n, dtype=float32)`: i × step + lower in float64, the last point set to upper,
        // then rounded to float32.
        func linspace32(_ l: (lower: Double, upper: Double), _ n: Int) -> [Float] {
            guard n > 1 else { return [Float(l.lower)] }
            let d = (l.upper - l.lower) / Double(n - 1)
            var out = (0..<n).map { Float(Double($0) * d + l.lower) }
            out[n - 1] = Float(l.upper)
            return out
        }
        let ax = linspace32(lx, nx), ay = linspace32(ly, ny)
        var radius = [Double](repeating: 0, count: nx * ny)
        for ix in 0..<nx {
            for iy in 0..<ny {
                let r: Float = (ax[ix] * ax[ix] + ay[iy] * ay[iy]).squareRoot()
                radius[ix * ny + iy] = Double(r)
            }
        }

        // Each pixel adds to the rings near its radius only (#61: was every ring over every pixel). A ring still sums its
        // pixels in index order with the same test and weight, so the result is bit for bit the full double loop's.
        var weights = [Double](repeating: 0, count: rings), sums = [Double](repeating: 0, count: rings)
        let gaussianScale = w * w / 2
        // exp(−x) is exactly 0 in float64 beyond x = 746, so those pixels add exactly 0; a non-finite value would add
        // NaN even there, so such a pattern keeps every ring.
        let reach = weighting == .step ? w : (746 * gaussianScale).squareRoot()
        let everyRing = weighting == .gaussian && !values.allSatisfy(\.isFinite)
        for k in radius.indices {
            let r = radius[k]
            // Held to the rings before Int, as `last` is: a pixel far past the last ring at a tiny step would not fit an Int (#129).
            // Past rings + 1 the pixel reaches no ring either way, so no sum changes.
            let first = everyRing ? 0 : Swift.max(0, Int(Swift.min(Double(rings) + 1, Swift.max(-1, ((r - reach) / step).rounded(.down)))) - 1)
            let last = everyRing ? rings - 1 : Swift.min(rings - 1, Int(Swift.min(Double(rings), ((r + reach) / step).rounded(.up))) + 1)
            guard first <= last else { continue }
            for i in first...last {
                let center = Double(i) * step
                switch weighting {
                case .step:
                    if abs(r - center) < w {
                        weights[i] += 1
                        sums[i] += values[k]
                    }
                case .gaussian:
                    let d = r - center
                    let m = exp(-(d * d) / gaussianScale)
                    weights[i] += m
                    sums[i] += values[k] * m
                }
            }
        }
        let out = (0..<rings).map { weights[$0] > 0 ? sums[$0] / weights[$0] : 0 }
        return try ReciprocalSpaceLineProfiles(count: rings, values: out, sampling: step / (wavelength * 1e3))
    }
}

extension ScanMeasurement {
    /// The azimuthal average of the pattern at every position of a `PixelatedDetector` scan, line or grid (abTEM
    /// `DiffractionPatterns.azimuthal_average` on a scan's patterns, whose scan axes become the profiles' ensemble axes):
    /// each position's pattern averaged as `RealField.azimuthalAverage(beam:maxAngle:radialSampling:weighting:width:)`
    /// averages it, bit for bit.
    /// - Parameters:
    ///   - grid: the grid of the scan's probe; the patterns are sampled at its reciprocal sampling (1/Å), as
    ///     `PixelatedDetector` crops them.
    ///   - beam: the scan's beam (α = k λ 10³ mrad).
    ///   - maxAngle, radialSampling, weighting, width: as `RealField.azimuthalAverage`.
    /// - Returns: profiles with `ensembleShape` the scan's shape, one per position in scan order.
    /// - Throws: `invalidScan` unless the measurement has two value axes no larger than the grid (a pixelated scan);
    ///   the refusals of `RealField.azimuthalAverage`.
    public func azimuthalAverage(grid: Grid, beam: ElectronBeam, maxAngle: Double? = nil, radialSampling: Double = 1,
                                 weighting: AzimuthalWeighting = .step, width: Double = 1) throws
        -> ReciprocalSpaceLineProfiles {
        guard valueShape.count == 2, valueShape[0] >= 1, valueShape[1] >= 1,
              valueShape[0] <= grid.nx, valueShape[1] <= grid.ny else {
            throw SimulatoError.invalidScan("an azimuthal average needs the patterns of a pixelated scan on \(grid.nx) x \(grid.ny); this one has value shape \(valueShape)")
        }
        let (nx, ny) = (valueShape[0], valueShape[1])
        let n = nx * ny, positions = values.count / n
        func average(_ pattern: [Double]) throws -> ReciprocalSpaceLineProfiles {
            try RealField(nx: nx, ny: ny, values: pattern, samplingX: grid.reciprocalSamplingX,
                          samplingY: grid.reciprocalSamplingY)
                .azimuthalAverage(beam: beam, maxAngle: maxAngle, radialSampling: radialSampling, weighting: weighting,
                                  width: width)
        }
        var out: [Double] = []
        var first: ReciprocalSpaceLineProfiles?
        for p in 0..<positions {
            let a = try average(Array(values[(p * n)..<((p + 1) * n)]))
            if first == nil {
                first = a
                out.reserveCapacity(positions * a.count)
            }
            out.append(contentsOf: a.values)
        }
        // With no position the rings still follow from the geometry alone.
        let rings = try first ?? average([Double](repeating: 0, count: n))
        return try ReciprocalSpaceLineProfiles(ensembleShape: scanShape, count: rings.count, values: out,
                                               sampling: rings.sampling)
    }
}

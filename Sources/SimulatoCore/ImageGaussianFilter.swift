//
//  ImageGaussianFilter.swift
//  Role: abTEM's `Images.gaussian_filter(sigma, boundary, cval)` (#105): σ in Å per axis, divided by the sampling, then
//        scipy.ndimage `gaussian_filter` as abTEM calls it: per axis with σ > 1e−15, axis 0 (x) first, a normalised kernel
//        exp(−x² / 2σ²) of radius int(4σ + 0.5) (truncate 4), correlated with the image extended by the boundary rule
//        (scipy's symmetric-kernel sum, centre first). CPU float64. Rows and columns: x is the first index, `[ix * ny + iy]`.
//

import Foundation

extension RealField {
    /// How a filter extends the image past its edges (abTEM's `boundary`).
    public enum FilterBoundary: Sendable, Hashable {
        /// Wrapped around to the opposite edge (abTEM's default, scipy's `wrap`): right for a periodic simulation cell.
        case periodic
        /// Mirrored about the edge of the last pixel (scipy's `reflect`: d c b a | a b c d).
        case reflect
        /// Filled with a constant value (scipy's `constant` with `cval`).
        case constant(Double)
    }

    /// The image blurred by a Gaussian of standard deviation `sigmaX`, `sigmaY` [Å] (abTEM `Images.gaussian_filter`), for
    /// example the partial spatial coherence, vibration or detector blur a simulated HRTEM image is given before it is set
    /// beside a measured one. A σ of 0 leaves that axis unchanged. The sampling stays.
    /// - Throws: `ImageOperationError.invalidSigma` for a σ that is negative or not finite, or whose kernel radius exceeds 2^23
    ///   pixels; for a constant boundary whose value is not finite.
    public func gaussianFiltered(sigmaX: Double, sigmaY: Double, boundary: FilterBoundary = .periodic) throws -> RealField {
        guard sigmaX.isFinite, sigmaY.isFinite, sigmaX >= 0, sigmaY >= 0 else {
            throw ImageOperationError.invalidSigma(sigmaX, sigmaY)
        }
        if case .constant(let c) = boundary, !c.isFinite { throw ImageOperationError.invalidSigma(sigmaX, sigmaY) }
        let px = sigmaX / samplingX, py = sigmaY / samplingY
        // A kernel radius above 2^23 pixels (as `gaussianSourceSize` refuses) would only allocate: every value is the mean.
        guard 4 * px + 0.5 <= Double(1 << 23), 4 * py + 0.5 <= Double(1 << 23) else {
            throw ImageOperationError.invalidSigma(sigmaX, sigmaY)
        }
        var out = values
        if px > 1e-15 { out = Self.correlate(out, count: nx, stride: ny, lines: ny, lineStride: 1, sigma: px, boundary: boundary) }
        if py > 1e-15 { out = Self.correlate(out, count: ny, stride: 1, lines: nx, lineStride: ny, sigma: py, boundary: boundary) }
        return RealField.unchecked(nx: nx, ny: ny, values: out, samplingX: samplingX, samplingY: samplingY)
    }

    /// scipy's `_gaussian_kernel1d(sigma, 0, radius)`: exp(−0.5 / σ² · x²) for x in −r…r, divided by its sum.
    static func gaussianKernel(sigma: Double) -> [Double] {
        let radius = Int(4.0 * sigma + 0.5)
        let s2 = sigma * sigma
        var w = (-radius...radius).map { x in exp(-0.5 / s2 * Double(x * x)) }
        let sum = w.reduce(0, +)
        for i in w.indices { w[i] /= sum }
        return w
    }

    /// One pass of scipy's `correlate1d` along an axis: `lines` lines of `count` values, element k of line l at
    /// `l * lineStride + k * stride`.
    private static func correlate(_ input: [Double], count n: Int, stride: Int, lines: Int, lineStride: Int, sigma: Double,
                                  boundary: FilterBoundary) -> [Double] {
        let w = gaussianKernel(sigma: sigma)
        let r = w.count / 2
        var out = input
        var line = [Double](repeating: 0, count: n + 2 * r)   // the line extended by r on each side
        for l in 0..<lines {
            let base = l * lineStride
            for e in -r..<(n + r) {
                let value: Double
                switch boundary {
                case .periodic:
                    value = input[base + (((e % n) + n) % n) * stride]
                case .reflect:
                    let m = ((e % (2 * n)) + 2 * n) % (2 * n)
                    value = input[base + (m < n ? m : 2 * n - 1 - m) * stride]
                case .constant(let c):
                    value = e >= 0 && e < n ? input[base + e * stride] : c
                }
                line[e + r] = value
            }
            // scipy's symmetric kernel: centre weight first, then each pair of mirrored samples.
            for k in 0..<n {
                let c = k + r
                var t = line[c] * w[r]
                if r > 0 {
                    for j in 1...r { t += (line[c - j] + line[c + j]) * w[r - j] }
                }
                out[base + k * stride] = t
            }
        }
        return out
    }
}

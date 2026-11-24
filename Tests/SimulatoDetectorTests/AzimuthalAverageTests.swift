//
//  AzimuthalAverageTests.swift
//  Role: the azimuthal average of a centred diffraction pattern (abTEM's DiffractionPatterns.azimuthal_average), against
//        numbers abTEM 1.0.10 printed (tools/pins/azimuthal/pins.py) for planted 9 x 8 and 8 x 6 patterns, values
//        (k + 1)^1.3 with k the flat index, at 200 keV; and its refusals.
//

import Foundation
import Testing
import SimulatoCore
import SimulatoDetector

@Suite("Azimuthal average")
struct AzimuthalAverageTests {
    static func pattern(_ nx: Int, _ ny: Int, _ sx: Double, _ sy: Double) -> RealField {
        RealField.unchecked(nx: nx, ny: ny, values: (0..<(nx * ny)).map { pow(Double($0) + 1, 1.3) }, samplingX: sx, samplingY: sy)
    }

    static let beam = try! ElectronBeam(energy: 200e3)

    /// assert_allclose defaults (rtol 1e-7, atol 0): abTEM states no tolerance for the azimuthal average.
    static func close(_ a: [Double], _ b: [Double], rtol: Double = 1e-7) -> Bool {
        a.count == b.count && zip(a, b).allSatisfy { abs($0 - $1) <= rtol * abs($1) }
    }

    @Test func stepRingsMatchAbTEMOnAnOddEvenPattern() throws {
        let p = Self.pattern(9, 8, 0.05, 0.06)
        let a = try p.azimuthalAverage(beam: Self.beam)
        #expect(a.count == 5 && a.ensembleShape == [])
        #expect(abs(a.sampling - 0.05000000000000001) <= 1e-7 * 0.05)
        #expect(Self.close(a.values, [109.31059072723754, 110.0733829078507, 111.05800473544934, 113.83482973524923,
                                      116.75029096542004]))
        let b = try p.azimuthalAverage(beam: Self.beam, maxAngle: 9, radialSampling: 0.7, width: 1.5)
        #expect(b.count == 11 && abs(b.sampling - 0.035) <= 1e-7 * 0.035)
        #expect(Self.close(b.values, [109.71729296135845, 110.60442943790507, 110.97479454458214, 112.91164715276787,
                                      115.2149506722666, 115.6100936043095, 116.37171872576145, 116.20544576168864,
                                      116.3659299808196, 111.20251111350319, 109.37593155122985]))
    }

    /// A step far below the pattern's angles: pixels lie some 1e20 steps out, which no Int holds; they reach no ring (#129).
    @Test func aTinyRadialSamplingDoesNotTrap() throws {
        let p = Self.pattern(9, 8, 0.05, 0.06)
        for weighting in [AzimuthalWeighting.step, .gaussian] {
            let a = try p.azimuthalAverage(beam: Self.beam, maxAngle: 1e-21, radialSampling: 1e-20, weighting: weighting)
            #expect(a.count == 1)
        }
    }

    @Test func gaussianRingsMatchAbTEM() throws {
        let p = Self.pattern(9, 8, 0.05, 0.06)
        let a = try p.azimuthalAverage(beam: Self.beam, weighting: .gaussian)
        #expect(Self.close(a.values, [109.52747242355026, 110.19514603212777, 111.79352436215716, 114.5200943034101,
                                      117.77733984988313]))
        let b = try p.azimuthalAverage(beam: Self.beam, maxAngle: 6, radialSampling: 1.3, weighting: .gaussian, width: 0.8)
        #expect(b.count == 4 && abs(b.sampling - 0.06500000000000002) <= 1e-7 * 0.065)
        #expect(Self.close(b.values, [109.39182869735909, 110.29310407876528, 112.87926982867883, 117.66284648473913]))
    }

    @Test func evenPatternMatchesAbTEM() throws {
        let p = Self.pattern(8, 6, 0.04, 0.04)
        let a = try p.azimuthalAverage(beam: Self.beam)
        #expect(a.count == 4 && abs(a.sampling - 0.039999999999999994) <= 1e-7 * 0.04)
        #expect(Self.close(a.values, [76.36746409305131, 76.84996447773611, 77.6798227805934, 75.22480102040664]))
        #expect(Self.close(try p.azimuthalAverage(beam: Self.beam, width: 0.5).values,
                           [76.08612050106568, 76.6139088690763, 77.74902662196723, 77.65019303476393]))
    }

    @Test func aRingWithNoPixelIsZero() throws {
        // A 1 x 1 pattern: one pixel at the centre; rings beyond a step width of it hold nothing.
        let p = RealField.unchecked(nx: 1, ny: 1, values: [3], samplingX: 0.05, samplingY: 0.05)
        let a = try p.azimuthalAverage(beam: Self.beam, maxAngle: 5, width: 0.5)
        #expect(a.count == 4 && a.values == [3, 0, 0, 0])
    }

    @Test func invalidArgumentsAreRefused() {
        let p = Self.pattern(9, 8, 0.05, 0.06)
        #expect(throws: SimulatoError.self) { try p.azimuthalAverage(beam: Self.beam, radialSampling: 0) }
        #expect(throws: SimulatoError.self) { try p.azimuthalAverage(beam: Self.beam, radialSampling: .nan) }
        #expect(throws: SimulatoError.self) { try p.azimuthalAverage(beam: Self.beam, width: 0) }
        #expect(throws: SimulatoError.self) { try p.azimuthalAverage(beam: Self.beam, width: -1) }
        #expect(throws: SimulatoError.self) { try p.azimuthalAverage(beam: Self.beam, maxAngle: 0) }
        #expect(throws: SimulatoError.self) { try p.azimuthalAverage(beam: Self.beam, maxAngle: .infinity) }
        #expect(throws: SimulatoError.self) { try p.azimuthalAverage(beam: Self.beam, maxAngle: 1e9) }
    }

    /// The average as it was before #61, every ring over every pixel: the per-pixel rings must equal it bit for bit.
    static func everyRingOverEveryPixel(_ p: RealField, maxAngle: Double?, radialSampling: Double,
                                        weighting: AzimuthalWeighting, width: Double) -> [Double] {
        let wavelength = Self.beam.wavelength
        func limits(_ n: Int, _ s: Double) -> (lower: Double, upper: Double) {
            let l = n % 2 == 1 ? (Double(-((n - 1) / 2)) * s, Double((n - 1) / 2) * s)
                               : (Double(-(n / 2)) * s, Double(n / 2 - 1) * s)
            return (l.0 * wavelength * 1e3, l.1 * wavelength * 1e3)
        }
        let lx = limits(p.nx, p.samplingX), ly = limits(p.ny, p.samplingY)
        let minAngular = min(p.samplingX * wavelength * 1e3, p.samplingY * wavelength * 1e3)
        let maximum = maxAngle ?? -min(lx.lower, ly.lower)
        let step = radialSampling * minAngular, w = width * minAngular
        let rings = max(Int((maximum / step).rounded(.up)), 0)
        func linspace32(_ l: (lower: Double, upper: Double), _ n: Int) -> [Float] {
            guard n > 1 else { return [Float(l.lower)] }
            let d = (l.upper - l.lower) / Double(n - 1)
            var out = (0..<n).map { Float(Double($0) * d + l.lower) }
            out[n - 1] = Float(l.upper)
            return out
        }
        let ax = linspace32(lx, p.nx), ay = linspace32(ly, p.ny)
        var radius = [Double](repeating: 0, count: p.nx * p.ny)
        for ix in 0..<p.nx {
            for iy in 0..<p.ny {
                let r: Float = (ax[ix] * ax[ix] + ay[iy] * ay[iy]).squareRoot()
                radius[ix * p.ny + iy] = Double(r)
            }
        }
        return (0..<rings).map { i in
            let center = Double(i) * step
            var weight = 0.0, sum = 0.0
            for k in radius.indices {
                switch weighting {
                case .step:
                    if abs(radius[k] - center) < w {
                        weight += 1
                        sum += p.values[k]
                    }
                case .gaussian:
                    let d = radius[k] - center
                    let m = exp(-(d * d) / (w * w / 2))
                    weight += m
                    sum += p.values[k] * m
                }
            }
            return weight > 0 ? sum / weight : 0
        }
    }

    /// (radial sampling, width, maximum angle) cases.
    static let ringCases: [(Double, Double, Double?)] = [
        (1.0, 1.0, nil), (0.7, 1.5, 30.0), (2.3, 0.4, nil), (1.0, 12.0, 40.0), (0.5, 3.0, nil),
    ]

    @Test func perPixelRingsEqualEveryRingOverEveryPixelBitForBit() throws {
        // A 64 x 48 pattern with signed values, anisotropic sampling.
        let values: [Double] = (0..<(64 * 48)).map { (k: Int) -> Double in
            let phase = sin(Double(k) * 0.37)
            return phase * Double(k % 17) - 3
        }
        let p = RealField.unchecked(nx: 64, ny: 48, values: values, samplingX: 0.05, samplingY: 0.061)
        for weighting in [AzimuthalWeighting.step, .gaussian] {
            for (sampling, width, maxAngle) in Self.ringCases {
                let a = try p.azimuthalAverage(beam: Self.beam, maxAngle: maxAngle, radialSampling: sampling,
                                               weighting: weighting, width: width)
                let b = Self.everyRingOverEveryPixel(p, maxAngle: maxAngle, radialSampling: sampling,
                                                     weighting: weighting, width: width)
                #expect(a.values == b, "\(weighting) sampling \(sampling) width \(width)")
            }
        }
        // A non-finite value reaches every Gaussian ring, as before.
        var bad = values
        bad[5] = .nan
        let q = RealField.unchecked(nx: 64, ny: 48, values: bad, samplingX: 0.05, samplingY: 0.061)
        let g = try q.azimuthalAverage(beam: Self.beam, weighting: .gaussian)
        let allNaN = g.values.allSatisfy { $0.isNaN }
        #expect(allNaN)
    }
}

//
//  ComplexFieldVectorTests.swift
//  Role: the complex field's vDSP helpers (#61) against the scalar formulas they replace: real scaling and real
//        weights exact, the complex product and |ψ|² within a few ulp of the largest magnitude (vDSP may fuse).
//

import Foundation
import Testing
import SimulatoCore

@Suite("Complex field on vDSP")
struct ComplexFieldVectorTests {
    static func field(_ seed: Double) throws -> ComplexField {
        let n = 37 * 23
        let re: [Double] = (0..<n).map { (k: Int) -> Double in sin(Double(k) * seed) * 3 }
        let im: [Double] = (0..<n).map { (k: Int) -> Double in cos(Double(k) * seed * 1.7) - 0.5 }
        return try ComplexField(nx: 37, ny: 23, real: re, imaginary: im)
    }

    @Test func scalingAndRealWeightsAreExact() throws {
        let f = try Self.field(0.31)
        var scaled = f
        scaled.scale(by: 1.0 / 851)
        #expect(scaled.real == f.real.map { $0 * (1.0 / 851) } && scaled.imaginary == f.imaginary.map { $0 * (1.0 / 851) })
        let w: [Double] = (0..<f.real.count).map { (k: Int) -> Double in Double(k % 7) / 7 }
        var weighted = f
        weighted.multiply(byReal: w)
        #expect(weighted.real == zip(f.real, w).map { $0 * $1 } && weighted.imaginary == zip(f.imaginary, w).map { $0 * $1 })
    }

    @Test func theComplexProductAndSquaredMagnitudeMatchTheFormulas() throws {
        let a = try Self.field(0.31), b = try Self.field(0.73)
        var product = a
        product.multiply(by: b)
        var worst = 0.0, scale = 0.0
        for i in a.real.indices {
            let re = a.real[i] * b.real[i] - a.imaginary[i] * b.imaginary[i]
            let im = a.real[i] * b.imaginary[i] + a.imaginary[i] * b.real[i]
            worst = max(worst, abs(product.real[i] - re), abs(product.imaginary[i] - im))
            scale = max(scale, abs(re), abs(im))
        }
        #expect(worst <= 4 * Double.ulpOfOne * scale, "\(worst) of \(scale)")
        let m = a.squaredMagnitude()
        let formula = a.real.indices.map { a.real[$0] * a.real[$0] + a.imaginary[$0] * a.imaginary[$0] }
        let mWorst = zip(m, formula).map { abs($0 - $1) }.max() ?? 0
        let mScale = formula.max() ?? 0
        #expect(mWorst <= 4 * Double.ulpOfOne * mScale, "\(mWorst) of \(mScale)")
    }
}

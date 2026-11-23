//
//  KirklandParameterization.swift
//  Role: Kirkland's (2010) parameterisation as abTEM's `KirklandParametrization`: the scaled parameters
//        a' = π a / κ, b' = 2π √b, c' = π^{3/2} c / d^{3/2} / κ, d' = π² / d, and the projected scattering factor of an
//        infinite projection, f(k²) = Σ [4π a' / (4π² k² + b'²) + √(π/d') c' π/d' exp(−π² k² / d')], three terms summed
//        in abTEM's order.
//

import Foundation
import SimulatoCore

/// Kirkland's electron scattering parameterisation (E. J. Kirkland, Advanced Computing in Electron Microscopy, 2nd ed.,
/// Springer 2010, Appendix C): three Lorentzians and three Gaussians, Z = 1 ... 103.
public struct KirklandParameterization: Sendable, Hashable {
    /// The π inside the squares of f(k²): `.exact` by default; `.abTEMFloat32` reproduces abTEM 1.0.10, which takes π² there
    /// as the float32 square of the float32 π (``float32PiSquared``) and the exact π elsewhere; `.exact` moves f by about 2e−8 of its maximum (π² itself by 6.7e−8).
    public typealias PiConstant = LobatoParameterization.PiConstant

    public var piConstant: PiConstant

    public init(piConstant: PiConstant = .exact) {
        self.piConstant = piConstant
    }

    /// Whether Z has parameters.
    public static func supports(atomicNumber z: Int) -> Bool {
        z >= 1 && z <= kirklandTable.count && kirklandTable[z - 1] != nil
    }

    /// The table's (a, b, c, d) for Z, as abTEM ships them.
    public static func parameters(atomicNumber z: Int) throws -> (a: [Double], b: [Double], c: [Double], d: [Double]) {
        guard supports(atomicNumber: z), let p = kirklandTable[z - 1] else { throw SimulatoError.unsupportedElement(z) }
        return p
    }

    /// The scaled parameters of the projected scattering factor (abTEM's `scaled_parameters`).
    public static func scaledParameters(atomicNumber z: Int) throws -> (a: [Double], b: [Double], c: [Double], d: [Double]) {
        let p = try parameters(atomicNumber: z)
        let kappa = PhysicalConstants.kappa
        let pi = Double.pi
        let pi32 = pow(pi, 1.5)
        var a: [Double] = [], b: [Double] = [], c: [Double] = [], d: [Double] = []
        for i in 0..<3 {
            a.append(pi * p.a[i] / kappa)
            b.append(2.0 * pi * p.b[i].squareRoot())
            c.append(pi32 * p.c[i] / pow(p.d[i], 1.5) / kappa)
            d.append(pi * pi / p.d[i])
        }
        return (a, b, c, d)
    }

    /// abTEM's π² in f: the float32 square of the float32 π, 9.86960506439209.
    public static let float32PiSquared: Double = {
        let pi = Float(Double.pi)
        return Double(pi * pi)
    }()

    /// f(k²) [V·Å²] for one element at the squared spatial frequencies k² [1/Å²].
    public func projectedScatteringFactor(atomicNumber z: Int, k2: [Double]) throws -> [Double] {
        let s = try Self.scaledParameters(atomicNumber: z)
        let pi = Double.pi
        let pi2: Double
        switch piConstant {
        case .exact: pi2 = pi * pi
        case .abTEMFloat32: pi2 = Self.float32PiSquared
        }
        let fourPi = 4 * pi
        let fourPi2 = 4 * pi2
        var lorentz = [Double](repeating: 0, count: 3)
        var gauss = [Double](repeating: 0, count: 3)
        for i in 0..<3 {
            lorentz[i] = fourPi * s.a[i]
            gauss[i] = (pi / s.d[i]).squareRoot() * s.c[i] * pi / s.d[i]
        }
        return k2.map { k in
            var sum = 0.0
            for i in 0..<3 {
                let b2 = s.b[i] * s.b[i]
                sum = sum + lorentz[i] / (fourPi2 * k + b2)
                sum = sum + gauss[i] * exp(-pi2 * k / s.d[i])
            }
            return sum
        }
    }

    /// f(k²) at one k².
    public func projectedScatteringFactor(atomicNumber z: Int, k2: Double) throws -> Double {
        try projectedScatteringFactor(atomicNumber: z, k2: [k2])[0]
    }
}

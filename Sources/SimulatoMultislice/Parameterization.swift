//
//  Parameterization.swift
//  Role: the choice of electron scattering parameterisation a potential is built from (abTEM's `parametrization=`).
//

import SimulatoCore

/// The electron scattering parameterisation of a potential: abTEM's `Potential(parametrization=)`.
public enum Parameterization: Sendable, Hashable {
    /// Lobato & Van Dyck (2014), abTEM's default.
    case lobato(LobatoParameterization)
    /// Kirkland (2010).
    case kirkland(KirklandParameterization)

    /// Lobato with the exact π, the kit's default.
    public static let standard = Parameterization.lobato(LobatoParameterization())

    /// Whether Z has parameters.
    public func supports(atomicNumber z: Int) -> Bool {
        switch self {
        case .lobato: return LobatoParameterization.supports(atomicNumber: z)
        case .kirkland: return KirklandParameterization.supports(atomicNumber: z)
        }
    }

    /// f(k²) [V·Å²] for one element at the squared spatial frequencies k² [1/Å²].
    public func projectedScatteringFactor(atomicNumber z: Int, k2: [Double]) throws -> [Double] {
        switch self {
        case .lobato(let p): return try p.projectedScatteringFactor(atomicNumber: z, k2: k2)
        case .kirkland(let p): return try p.projectedScatteringFactor(atomicNumber: z, k2: k2)
        }
    }

    /// f(k²) at one k².
    public func projectedScatteringFactor(atomicNumber z: Int, k2: Double) throws -> Double {
        try projectedScatteringFactor(atomicNumber: z, k2: [k2])[0]
    }
}

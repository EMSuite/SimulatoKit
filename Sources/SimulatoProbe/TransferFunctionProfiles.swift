//
//  TransferFunctionProfiles.swift
//  Role: abTEM's `CTF.profiles` (radial line profiles of the transfer function and of each factor) and the CTF's own
//        Scherzer defocus, point resolution and crossover angle (`CTF.scherzer_defocus`, `.point_resolution`,
//        `.crossover_angle`), for the transfer function strip of an HRTEM view (#102).
//

import Foundation
import SimulatoCore

/// Radial line profiles of a contrast transfer function at one azimuth (abTEM's `CTF.profiles`).
///
/// Every array has one value per angle in `alpha`. A factor whose parameter is unset (no angular spread, no focal
/// spread, no aperture) is `nil`, as abTEM leaves its row out.
public struct TransferFunctionProfiles: Sendable, Hashable {
    /// The scattering angles α [rad]: NumPy's `linspace(0, maxAngle · 10⁻³, points)`.
    public let alpha: [Double]
    /// The largest angle [mrad].
    public let maxAngle: Double
    /// The step in spatial frequency [1/Å]: maxAngle / (points − 1) / (λ · 10³), abTEM's `sampling`.
    public let sampling: Double
    /// The wavelength λ [Å] the profiles were taken at.
    public let wavelength: Double
    /// The azimuth φ [rad].
    public let phi: Double
    /// The phase term alone, Im exp(−iχ) = −sin χ.
    public let phase: [Double]
    /// The spatial (partial spatial coherence) envelope, when the angular spread is not zero.
    public let spatialEnvelope: [Double]?
    /// The temporal (focal spread) envelope, when the focal spread is not zero.
    public let temporalEnvelope: [Double]?
    /// The aperture, when the cutoff is finite. Its edge is hard, as abTEM's profiles have no grid to soften it on.
    public let aperture: [Double]?
    /// The transfer function: the product of `phase` and every factor present (abTEM's `ctf` row).
    public let transfer: [Double]

    /// The scattering angles [mrad], α · 10³.
    public var angles: [Double] { alpha.map { $0 * 1e3 } }
    /// The spatial frequencies [1/Å], α / λ.
    public var spatialFrequencies: [Double] { alpha.map { $0 / wavelength } }
}

extension ContrastTransferFunction {
    /// Radial line profiles of this transfer function and of each factor at azimuth `phi` (abTEM's `CTF.profiles`).
    ///
    /// - Parameters:
    ///   - energy: the electron energy [eV].
    ///   - points: the number of angles, at least 2 (abTEM's `gpts`, 1000 by default).
    ///   - maxAngle: the largest angle [mrad]; `nil` takes abTEM 1.0.10's default, 50 mrad without an aperture and
    ///     1.6 × the cutoff with one.
    ///   - phi: the azimuth [rad], 0 by default.
    /// - Throws: `SimulatoError.invalidEnergy`, or `invalidAberration` for fewer than 2 points, a non-finite or
    ///   non-positive `maxAngle`, or a non-finite `phi`.
    public func profiles(energy: Double, points: Int = 1000, maxAngle: Double? = nil, phi: Double = 0) throws
        -> TransferFunctionProfiles
    {
        let wavelength = try ElectronBeam(energy: energy).wavelength
        guard points >= 2 else { throw SimulatoError.invalidAberration("a profile needs at least 2 points, got \(points)") }
        let top = maxAngle ?? (semiangleCutoff.isFinite ? semiangleCutoff * 1.6 : 50)
        guard top.isFinite, top > 0 else {
            throw SimulatoError.invalidAberration("the largest angle must be finite and positive, got \(top) mrad")
        }
        guard phi.isFinite else { throw SimulatoError.invalidAberration("the azimuth must be finite, got \(phi) rad") }

        // NumPy's linspace(0, stop, n): i · (stop / (n − 1)), the last value exactly stop.
        let stop = top * 1e-3
        let step = stop / Double(points - 1)
        var alpha = (0..<points).map { Double($0) * step }
        alpha[points - 1] = stop

        let phase = alpha.map { -sin(aberrations.chi(alpha: $0, phi: phi, wavelength: wavelength)) }
        let spatial = angularSpread != 0
            ? alpha.map { Self.spatialEnvelope(alpha: $0, phi: phi, wavelength: wavelength, aberrations: aberrations,
                                               angularSpread: angularSpread) }
            : nil
        let temporal = focalSpread != 0
            ? alpha.map { Self.temporalEnvelope(alpha: $0, wavelength: wavelength, focalSpread: focalSpread) }
            : nil
        let cutoff = semiangleCutoff * 1e-3
        let aperture = semiangleCutoff.isFinite ? alpha.map { $0 <= cutoff ? 1.0 : 0.0 } : nil

        // abTEM multiplies in its row order: phase, spatial, temporal, aperture.
        var transfer = phase
        for factor in [spatial, temporal, aperture] {
            guard let factor else { continue }
            for i in transfer.indices { transfer[i] *= factor[i] }
        }
        return TransferFunctionProfiles(alpha: alpha, maxAngle: top, sampling: top / Double(points - 1) / (wavelength * 1e3),
                                        wavelength: wavelength, phi: phi, phase: phase, spatialEnvelope: spatial,
                                        temporalEnvelope: temporal, aperture: aperture, transfer: transfer)
    }

    /// The Scherzer defocus [Å] for this function's own Cs (abTEM's `CTF.scherzer_defocus`), with abTEM's sign
    /// (defocus = −C10) and factor 1.5.
    /// - Throws: `SimulatoError.invalidAberration` when Cs is 0, as abTEM refuses it; `invalidEnergy`.
    public func scherzerDefocus(energy: Double) throws -> Double {
        guard aberrations.Cs != 0 else {
            throw SimulatoError.invalidAberration("Scherzer defocus needs a non-zero Cs")
        }
        return try Self.scherzerDefocus(Cs: aberrations.Cs, energy: energy)
    }

    /// The Scherzer point resolution [Å] for this function's own Cs (abTEM's `CTF.point_resolution`).
    /// - Throws: `SimulatoError.invalidEnergy`.
    public func pointResolution(energy: Double) throws -> Double {
        try Self.pointResolution(Cs: aberrations.Cs, energy: energy)
    }

    /// The first zero crossing of the phase at Scherzer defocus [mrad], 10³ λ / point resolution (abTEM's
    /// `CTF.crossover_angle`).
    /// - Throws: `SimulatoError.invalidAberration` when Cs is 0 (abTEM divides by zero); `invalidEnergy`.
    public func crossoverAngle(energy: Double) throws -> Double {
        guard aberrations.Cs != 0 else {
            throw SimulatoError.invalidAberration("the crossover angle needs a non-zero Cs")
        }
        let wavelength = try ElectronBeam(energy: energy).wavelength
        let resolution = try pointResolution(energy: energy)
        return 1e3 * wavelength / resolution
    }
}

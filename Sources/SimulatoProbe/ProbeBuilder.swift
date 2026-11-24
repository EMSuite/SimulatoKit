//
//  ProbeBuilder.swift
//  Role: builds one probe at many positions: the aperture and the aberration phase exp(−iχ) over the grid, the
//        per-column shift factors and the FFT plan are computed once, and each position does only the shift, the
//        norm and the inverse FFT, in the order `Probe.build(at:)` uses, so every wave is bit for bit the same (#61).
//

import Foundation
import SimulatoCore

/// A probe's position-independent work, done once; ``build(at:)`` equals `Probe.build(at:)` bit for bit.
/// Not for concurrent use (it holds an FFT plan).
package struct ProbeBuilder {
    package let probe: Probe
    private let aperture: [Double]
    private let cosChi: [Double]
    private let sinChi: [Double]
    private let px: [Double]
    private let py: [Double]
    private let fft: FFT2D

    package init(_ probe: Probe) throws {
        self.probe = probe
        WorkCounter.current?.countProbeSpectrum()
        let grid = probe.grid, lambda = probe.beam.wavelength
        let kx = grid.frequenciesX, ky = grid.frequenciesY
        aperture = probe.aperture.values(on: grid, wavelength: lambda)
        var c = [Double](repeating: 0, count: grid.count), s = c
        for ix in 0..<grid.nx {
            for iy in 0..<grid.ny {
                let i = ix * grid.ny + iy
                let alpha = (kx[ix] * kx[ix] + ky[iy] * ky[iy]).squareRoot() * lambda
                let phi = atan2(ky[iy], kx[ix])
                let chi = probe.aberrations.chi(alpha: alpha, phi: phi, wavelength: lambda)
                c[i] = cos(-chi)
                s[i] = sin(-chi)
            }
        }
        cosChi = c
        sinChi = s
        px = Grid.fftfreq(grid.nx, 1.0)
        py = Grid.fftfreq(grid.ny, 1.0)
        fft = try FFT2D(nx: grid.nx, ny: grid.ny)
    }

    /// The probe centred at `position` (x, y) [Å], as `Probe.build(at:)` returns it.
    package func build(at position: SIMD2<Double>) throws -> Waves {
        guard position.x.isFinite, position.y.isFinite else { throw SimulatoError.invalidPosition(index: 0) }
        let grid = probe.grid
        let ux = position.x / grid.samplingX, uy = position.y / grid.samplingY
        var cyAll = [Double](repeating: 0, count: grid.ny), syAll = cyAll
        for iy in 0..<grid.ny {
            let sy = -2 * Double.pi * py[iy] * uy
            (cyAll[iy], syAll[iy]) = (cos(sy), sin(sy))
        }
        var psi = ComplexField(nx: grid.nx, ny: grid.ny)
        for ix in 0..<grid.nx {
            let sx = -2 * Double.pi * px[ix] * ux
            let (cx, snx) = (cos(sx), sin(sx))
            for iy in 0..<grid.ny {
                let i = ix * grid.ny + iy
                let (cy, sny) = (cyAll[iy], syAll[iy])
                // shift = e^{i sx} · e^{i sy}
                var re = cx * cy - snx * sny
                var im = cx * sny + snx * cy
                re *= aperture[i]
                im *= aperture[i]
                let (c, s) = (cosChi[i], sinChi[i])
                psi.real[i] = re * c - im * s
                psi.imaginary[i] = re * s + im * c
            }
        }
        var norm = 0.0
        for i in psi.real.indices { norm += psi.real[i] * psi.real[i] + psi.imaginary[i] * psi.imaginary[i] }
        let f = norm.squareRoot()
        for i in psi.real.indices {
            psi.real[i] /= f
            psi.imaginary[i] /= f
        }
        fft.inverse(&psi)
        return Waves.unchecked(field: psi, grid: grid, beam: probe.beam, normalization: .reciprocalSpace,
                               tilt: probe.tilt)
    }
}

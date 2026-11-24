//
//  HRTEMSeries.swift
//  Role: HRTEM images over a list of defocus values for exit waves stored at several thicknesses (abTEM
//        `waves.apply_ctf(CTF(defocus=…)).intensity()` for each defocus, then `Images.tile`): each exit wave is
//        Fourier-transformed once and each defocus kernel built once, and every image equals `Waves.applying(_:)` then
//        `intensity()` with that defocus, bit for bit (#88); on `.metal` the products, inverse transforms and
//        intensities run on the GPU in float32 (#89).
//

import Foundation
import SimulatoCore
import SimulatoMultislice

extension ContrastTransferFunction {
    /// The HRTEM images of `waves` (for example the exit waves of `Multislice.run(_:through:exitPlanes:)`, one per
    /// thickness) at each of `defocus` values [Å, abTEM's sign: defocus = −C10], with every other setting of this
    /// function: `images[w][d]` is wave `w` at `defocus[d]`, tiled `tile.x` × `tile.y` times.
    ///
    /// Each wave is transformed once and each kernel built once per grid and energy, so a series costs one inverse
    /// transform per image. Every image equals `waves[w].applying(ctf).intensity()` for this function with
    /// `aberrations.defocus = defocus[d]`, bit for bit, then `tiled(repetitionsX:repetitionsY:)`.
    ///
    /// On `.metal` the kernels are still computed in float64 on the CPU; the transforms, products and intensities run in
    /// float32 on the GPU, batched by `options.metalBatchBytes`, within `MetalAccuracy.registered.intensity` × each
    /// image's maximum of `.cpu`.
    /// - Throws: `SimulatoError.invalidAberration` for a non-finite defocus, `ImageOperationError.invalidRepetitions`
    ///   for a tile count below 1 or an image too large, and on `.metal` `SimulatoError.metalUnavailable` without a
    ///   Metal device.
    public func images(of waves: [Waves], defocus: [Double], tile: (x: Int, y: Int) = (1, 1), backend: Backend = .cpu,
                       options: ExecutionOptions = .standard) throws -> [[RealField]] {
        if let bad = defocus.first(where: { !$0.isFinite }) {
            throw SimulatoError.invalidAberration("every defocus must be finite, got \(bad) Å")
        }
        guard tile.x >= 1, tile.y >= 1 else { throw ImageOperationError.invalidRepetitions(tile.x, tile.y) }
        struct Key: Hashable { let grid: Grid; let energy: Double }
        var kernels: [Key: [ComplexField]] = [:]
        var plans: [Grid: FFT2D] = [:]
        var out: [[RealField]] = []
        out.reserveCapacity(waves.count)
        for wave in waves {
            let key = Key(grid: wave.grid, energy: wave.beam.energy)
            if kernels[key] == nil {
                kernels[key] = try defocus.map { d in
                    var ctf = self
                    ctf.aberrations.defocus = d
                    return try ctf.kernel(on: wave.grid, wavelength: wave.beam.wavelength)
                }
            }
            if backend == .metal {
                let values = try Multislice.metalCTFIntensities(wave.field, kernels: kernels[key]!, grid: wave.grid,
                                                                options: options)
                out.append(try values.map { v in
                    let image = RealField.unchecked(nx: wave.grid.nx, ny: wave.grid.ny, values: v,
                                                    samplingX: wave.grid.samplingX, samplingY: wave.grid.samplingY)
                    if tile.x == 1 && tile.y == 1 { return image }
                    return try image.tiled(repetitionsX: tile.x, repetitionsY: tile.y)
                })
                continue
            }
            if plans[wave.grid] == nil { plans[wave.grid] = try FFT2D(nx: wave.grid.nx, ny: wave.grid.ny) }
            let fft = plans[wave.grid]!
            var spectrum = wave.field
            fft.forward(&spectrum)
            var row: [RealField] = []
            row.reserveCapacity(defocus.count)
            for h in kernels[key]! {
                // The product and its inverse exactly as `Waves.applying(_:)` computes them.
                var f = spectrum
                for i in f.real.indices {
                    let a = f.real[i], b = f.imaginary[i], c = h.real[i], d = h.imaginary[i]
                    f.real[i] = a * c - b * d
                    f.imaginary[i] = a * d + b * c
                }
                fft.inverse(&f)
                let image = Waves.unchecked(field: f, grid: wave.grid, beam: wave.beam, normalization: wave.normalization,
                                            tilt: wave.tilt).intensity()
                if tile.x == 1 && tile.y == 1 {
                    row.append(image)
                } else {
                    row.append(try image.tiled(repetitionsX: tile.x, repetitionsY: tile.y))
                }
            }
            out.append(row)
        }
        return out
    }
}

//
//  MetalCTFImages.swift
//  Role: the HRTEM image series on Metal (#89): an exit wave's spectrum times a batch of CTF kernels, the inverse FFT
//        and |ψ|², one executable per grid and batch, read back widened to float64.
//

import Foundation
import Metal
import MetalPerformanceShadersGraph
import SimulatoCore

extension MetalMultislice {
    /// |IFFT(FFT(ψ) · H_d)|² for each kernel H_d, in float32 on the GPU, in batches of
    /// `options.metalBatchBytes` (the kernels come computed in float64 and are cast).
    static func ctfIntensities(_ wave: ComplexField, kernels: [ComplexField], grid: Grid,
                               options: ExecutionOptions = .standard) throws -> [[Double]] {
        let engine = try MetalEngine.require()
        let nx = grid.nx, ny = grid.ny, n = grid.count
        let psi = try engine.complex(wave)
        var out: [[Double]] = []
        out.reserveCapacity(kernels.count)
        for (start, used, b) in batchPlan(count: kernels.count, grid: grid, options: options) {
            try autoreleasepool {
                let step = engine.locked {
                    engine.step("ctf images \(nx)x\(ny) b\(b)", limit: options.metalExecutableCacheLimit) {
                        let g = MPSGraph()
                        let w = g.placeholder(shape: MetalGraphs.shape([nx, ny]), dataType: .complexFloat32, name: "psi")
                        let h = g.placeholder(shape: MetalGraphs.shape([b, nx, ny]), dataType: .complexFloat32, name: "h")
                        let f = g.fastFourierTransform(w, axes: [0, 1], descriptor: MetalGraphs.forward, name: nil)
                        let product = g.multiplication(h, g.reshape(f, shape: MetalGraphs.shape([1, nx, ny]), name: nil),
                                                       name: nil)
                        let image = g.fastFourierTransform(product, axes: [1, 2], descriptor: MetalGraphs.inverse, name: nil)
                        let re = g.realPartOfTensor(tensor: image, name: nil)
                        let im = g.imaginaryPartOfTensor(tensor: image, name: nil)
                        let out = g.addition(g.multiplication(re, re, name: nil), g.multiplication(im, im, name: nil),
                                             name: nil)
                        return MetalStep(executable: MetalExecutable(g, device: engine.graphDevice, feeds: [
                            (w, [nx, ny], .complexFloat32), (h, [b, nx, ny], .complexFloat32),
                        ], targets: [out]), placeholders: ["psi": w, "h": h])
                    }
                }
                // The padded kernels stay zero: zero images, not read.
                var h = [Float](repeating: 0, count: 2 * b * n)
                for k in 0..<used {
                    let kernel = kernels[start + k]
                    for i in 0..<n {
                        h[2 * (k * n + i)] = Float(kernel.real[i])
                        h[2 * (k * n + i) + 1] = Float(kernel.imaginary[i])
                    }
                }
                let hData = try engine.floats(h, [b, nx, ny], .complexFloat32)
                let (values, buffer) = try engine.tensor([b, nx, ny], .float32)
                try engine.locked {
                    let cb = MPSCommandBuffer(from: engine.queue)
                    _ = step.executable.executable.encode(to: cb, inputs: step.inputs(["psi": psi, "h": hData]),
                                                          results: [values], executionDescriptor: nil)
                    cb.commit()
                    engine.notePeak()
                    cb.waitUntilCompleted()
                    try engine.check(cb, "CTF images")
                }
                let pointer = buffer.contents().bindMemory(to: Float.self, capacity: b * n)
                for k in 0..<used {
                    out.append((0..<n).map { Double(pointer[k * n + $0]) })
                }
            }
        }
        return out
    }
}

extension Multislice {
    /// The HRTEM images |IFFT(FFT(ψ) · H_d)|² of one wave's field for each CTF kernel H_d (float64, on the wave's grid),
    /// on Metal in float32, widened to float64; within `MetalAccuracy.registered.intensity` × the image's maximum of
    /// the CPU product. For `ContrastTransferFunction.images(of:defocus:tile:backend:)`.
    /// - Throws: `SimulatoError.metalUnavailable` without a Metal device or for a batch over the device's largest
    ///   buffer, `SimulatoError.metalFailure` when a command buffer fails.
    package static func metalCTFIntensities(_ wave: ComplexField, kernels: [ComplexField], grid: Grid,
                                            options: ExecutionOptions = .standard) throws -> [[Double]] {
        try MetalMultislice.ctfIntensities(wave, kernels: kernels, grid: grid, options: options)
    }
}

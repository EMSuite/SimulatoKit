//
//  Fields.swift
//  Role: the arrays the kit returns: a complex field (split real and imaginary parts, vDSP's layout) and a real field,
//        both on a grid, x the first index, row-major `[ix * ny + iy]` (abTEM's `[x, y]`).
//

import Accelerate

/// A complex array on an nx × ny grid, split into real and imaginary parts.
public struct ComplexField: Sendable, Hashable {
    public let nx: Int
    public let ny: Int
    /// The real parts, nx × ny of them. Set by the kit only, so the count always matches the grid (a resized array
    /// would reach the FFT); build a new field to change them.
    public package(set) var real: [Double]
    /// The imaginary parts, nx × ny of them (set by the kit only, as `real`).
    public package(set) var imaginary: [Double]

    /// A field of zeros.
    public init(nx: Int, ny: Int) {
        self.nx = nx
        self.ny = ny
        real = [Double](repeating: 0, count: nx * ny)
        imaginary = [Double](repeating: 0, count: nx * ny)
    }

    /// - Throws: `SimulatoError.invalidShape` unless both arrays hold nx × ny values (nx and ny positive).
    public init(nx: Int, ny: Int, real: [Double], imaginary: [Double]) throws {
        guard Self.fills(nx, ny, real.count), imaginary.count == real.count else {
            throw SimulatoError.invalidShape("a \(nx) x \(ny) complex field needs \(nx) x \(ny) real and imaginary parts, got \(real.count) and \(imaginary.count)")
        }
        self.init(storing: nx, ny, real, imaginary)
    }

    private init(storing nx: Int, _ ny: Int, _ real: [Double], _ imaginary: [Double]) {
        self.nx = nx
        self.ny = ny
        self.real = real
        self.imaginary = imaginary
    }

    /// The kit's own construction, from arrays it sized: a mismatch is a bug of the kit and traps.
    package static func unchecked(nx: Int, ny: Int, real: [Double], imaginary: [Double]) -> ComplexField {
        precondition(real.count == nx * ny && imaginary.count == nx * ny, "ComplexField: \(nx) x \(ny) values required")
        return ComplexField(storing: nx, ny, real, imaginary)
    }

    /// Whether `count` values fill a positive nx × ny grid (false where nx · ny overflows).
    package static func fills(_ nx: Int, _ ny: Int, _ count: Int) -> Bool {
        let (product, overflow) = nx.multipliedReportingOverflow(by: ny)
        return nx > 0 && ny > 0 && !overflow && product == count
    }

    /// The value at (ix, iy).
    public subscript(ix: Int, iy: Int) -> (re: Double, im: Double) {
        (real[ix * ny + iy], imaginary[ix * ny + iy])
    }

    /// |ψ|² per pixel (abTEM's `abs2`: re² + im²).
    public func squaredMagnitude() -> [Double] {
        var out = [Double](repeating: 0, count: real.count)
        guard !out.isEmpty else { return out }
        real.withUnsafeBufferPointer { r in
            imaginary.withUnsafeBufferPointer { i in
                var split = DSPDoubleSplitComplex(realp: UnsafeMutablePointer(mutating: r.baseAddress!),
                                                  imagp: UnsafeMutablePointer(mutating: i.baseAddress!))
                vDSP_zvmagsD(&split, 1, &out, 1, vDSP_Length(out.count))
            }
        }
        return out
    }

    /// The field as vDSP's split complex, for `body` to change in place (#61: the kit's loops on vDSP). The field must
    /// not be empty.
    private mutating func withSplitComplex<R>(_ body: (inout DSPDoubleSplitComplex, vDSP_Length) throws -> R) rethrows -> R {
        let n = vDSP_Length(real.count)
        return try real.withUnsafeMutableBufferPointer { r in
            try imaginary.withUnsafeMutableBufferPointer { i in
                var split = DSPDoubleSplitComplex(realp: r.baseAddress!, imagp: i.baseAddress!)
                return try body(&split, n)
            }
        }
    }

    /// Every element times a real scale, in place.
    package mutating func scale(by s: Double) {
        guard !real.isEmpty else { return }
        var factor = s
        let n = vDSP_Length(real.count)
        real.withUnsafeMutableBufferPointer { vDSP_vsmulD($0.baseAddress!, 1, &factor, $0.baseAddress!, 1, n) }
        imaginary.withUnsafeMutableBufferPointer { vDSP_vsmulD($0.baseAddress!, 1, &factor, $0.baseAddress!, 1, n) }
    }

    /// Every element times the real `weights` (one per element), in place.
    package mutating func multiply(byReal weights: [Double]) {
        precondition(weights.count == real.count, "ComplexField: one weight per element required")
        guard !real.isEmpty else { return }
        let n = vDSP_Length(real.count)
        weights.withUnsafeBufferPointer { w in
            real.withUnsafeMutableBufferPointer { vDSP_vmulD($0.baseAddress!, 1, w.baseAddress!, 1, $0.baseAddress!, 1, n) }
            imaginary.withUnsafeMutableBufferPointer { vDSP_vmulD($0.baseAddress!, 1, w.baseAddress!, 1, $0.baseAddress!, 1, n) }
        }
    }

    /// Element-wise ψ ← ψ · t, in place (vDSP's complex multiply).
    package mutating func multiply(by t: ComplexField) {
        precondition(t.nx == nx && t.ny == ny, "ComplexField: shapes \(nx) x \(ny) and \(t.nx) x \(t.ny) differ")
        guard !real.isEmpty else { return }
        t.real.withUnsafeBufferPointer { tr in
            t.imaginary.withUnsafeBufferPointer { ti in
                var b = DSPDoubleSplitComplex(realp: UnsafeMutablePointer(mutating: tr.baseAddress!),
                                              imagp: UnsafeMutablePointer(mutating: ti.baseAddress!))
                withSplitComplex { a, n in vDSP_zvmulD(&a, 1, &b, 1, &a, 1, n, 1) }
            }
        }
    }

    /// The largest |ψ|.
    public var maxMagnitude: Double {
        var m = 0.0
        for i in real.indices { m = Swift.max(m, (real[i] * real[i] + imaginary[i] * imaginary[i]).squareRoot()) }
        return m
    }
}

/// A real array on an nx × ny grid with its sampling (Å per pixel for images, 1/Å per pixel for patterns).
public struct RealField: Sendable, Hashable {
    public let nx: Int
    public let ny: Int
    /// nx × ny values, row-major (set by the kit only, so the count always matches the grid).
    public package(set) var values: [Double]
    public let samplingX: Double
    public let samplingY: Double

    /// - Throws: `SimulatoError.invalidShape` unless `values` holds nx × ny numbers (nx and ny positive).
    public init(nx: Int, ny: Int, values: [Double], samplingX: Double, samplingY: Double) throws {
        guard ComplexField.fills(nx, ny, values.count) else {
            throw SimulatoError.invalidShape("a \(nx) x \(ny) real field needs \(nx) x \(ny) values, got \(values.count)")
        }
        self.init(storing: nx, ny, values, samplingX, samplingY)
    }

    /// The kit's own construction, from an array it sized: a mismatch is a bug of the kit and traps.
    package static func unchecked(nx: Int, ny: Int, values: [Double], samplingX: Double, samplingY: Double) -> RealField {
        precondition(values.count == nx * ny, "RealField: \(nx) x \(ny) values required")
        return RealField(storing: nx, ny, values, samplingX, samplingY)
    }

    private init(storing nx: Int, _ ny: Int, _ values: [Double], _ samplingX: Double, _ samplingY: Double) {
        self.nx = nx
        self.ny = ny
        self.values = values
        self.samplingX = samplingX
        self.samplingY = samplingY
    }

    public subscript(ix: Int, iy: Int) -> Double { values[ix * ny + iy] }

    public var sum: Double { values.reduce(0, +) }
    public var max: Double { values.max() ?? 0 }
}

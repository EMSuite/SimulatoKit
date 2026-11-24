//
//  ImageComparisonTests.swift
//  Role: the crop and the relative difference of images (abTEM's Images.crop and BaseMeasurements.relative_difference)
//        against what abTEM 1.0.10 gave in float64 (tools/pins/image-crop-difference/pins.py) for a planted 7 x 9 image at
//        sampling (0.3, 0.25) A, values sin(0.7 i) + cos(0.4 j)^2 + 0.05 i j + 2; their truths and refusals.
//

import Foundation
import Testing
import SimulatoCore

@Suite("Image crop and relative difference")
struct ImageComparisonTests {
    static func planted(_ f: (Int, Int) -> Double) -> RealField {
        var v = [Double]()
        for i in 0..<7 { for j in 0..<9 { v.append(f(i, j)) } }
        return RealField.unchecked(nx: 7, ny: 9, values: v, samplingX: 0.3, samplingY: 0.25)
    }

    static let a = planted { i, j in sin(0.7 * Double(i)) + pow(cos(0.4 * Double(j)), 2) + 0.05 * Double(i * j) + 2 }
    static let b = planted { i, j in a[i, j] * 1.01 + 0.02 * cos(Double(i)) }

    /// The window abTEM's crop takes, as (origin x, origin y, rows, columns): a slice of the image, so held exact.
    static let crops: [(String, Double, Double, Double, Double, Bool, Int, Int, Int, Int)] = [
        ("inside", 1.2, 1.0, 0.3, 0.5, false, 1, 2, 4, 4),
        ("half to even", 0.75, 0.625, 0.45, 0.125, false, 2, 0, 2, 2),
        ("cut at the edge", 1.2, 1.0, 1.8, 1.75, false, 6, 7, 1, 2),
        ("whole", 2.1, 2.25, 0, 0, false, 0, 0, 7, 9),
        ("centred", 1.0, 1.0, 0, 0, true, 2, 2, 3, 4),
    ]

    @Test("the crop is abTEM's window, exact", arguments: 0..<5)
    func cropMatchesAbTEM(_ k: Int) throws {
        let (name, ex, ey, ox, oy, centred, x0, y0, mx, my) = Self.crops[k]
        let c = try centred ? Self.a.croppedCentered(extentX: ex, extentY: ey)
                            : Self.a.cropped(extentX: ex, extentY: ey, offsetX: ox, offsetY: oy)
        #expect(c.nx == mx && c.ny == my, "\(name)")
        #expect(c.samplingX == 0.3 && c.samplingY == 0.25)
        for ix in 0..<mx { for iy in 0..<my { #expect(c[ix, iy] == Self.a[x0 + ix, y0 + iy], "\(name) (\(ix), \(iy))") } }
    }

    @Test("the crop refuses what abTEM refuses, and an empty or misplaced window")
    func cropRefusals() {
        let a = Self.a
        #expect(throws: ImageOperationError.self) { try a.cropped(extentX: 2.2, extentY: 1) }
        #expect(throws: ImageOperationError.self) { try a.cropped(extentX: 0, extentY: 1) }
        #expect(throws: ImageOperationError.self) { try a.cropped(extentX: .nan, extentY: 1) }
        #expect(throws: ImageOperationError.self) { try a.cropped(extentX: 1, extentY: 1, offsetX: -0.3) }
        #expect(throws: ImageOperationError.self) { try a.cropped(extentX: 1, extentY: 1, offsetX: .infinity) }
        #expect(throws: ImageOperationError.self) { try a.cropped(extentX: 1, extentY: 1, offsetX: 2.1) }
        #expect(throws: ImageOperationError.self) { try a.cropped(extentX: 0.1, extentY: 1) }
        #expect(throws: ImageOperationError.self) { try a.croppedCentered(extentX: 3, extentY: 1) }
    }

    /// abTEM's relative difference of a to b with min_relative_tol 0, row-major.
    static let difference: [Double] = [
        -1.6666666666666756, -1.7021600731940045, -1.8046993674250014, -1.9383930236728182,
        -1.9995738756080899, -1.9203110953351672, -1.786240943343787, -1.692572829076921,
        -1.6674247589925602, -1.2965258128021926, -1.305033999022728, -1.3345920899830888,
        -1.3693717032714345, -1.3798164888884719, -1.3522873000864635, -1.3098092832568766,
        -1.2783628122988075, -1.267422762398542, -0.7911669373642554, -0.7884251816418011,
        -0.7732694933980714, -0.7564079978210391, -0.7542175538347432, -0.7725120751838738,
        -0.7984370265501515, -0.8180075816628022, -0.8259543433253522, -0.48747665340269675,
        -0.4872581034957692, -0.4573316393682664, -0.42517700600666497, -0.4284210255630248,
        -0.4770780952018801, -0.5402823527432272, -0.5875883844333587, -0.6086832920973783,
        -0.6080084297552554, -0.6136106154636828, -0.594059137031283, -0.5736584958599009,
        -0.5831142775292985, -0.6273588207821282, -0.6794872932484058, -0.717207013619679,
        -0.7349151630616023, -1.2141479613356512, -1.2064822169067613, -1.2153346647817844,
        -1.2241928102073942, -1.2140790630854106, -1.1846521615761243, -1.1536229497822017,
        -1.1323359946575036, -1.1221152963097707, -1.9022358176585756, -1.8434467153328784,
        -1.8674312721787065, -1.8891587892709383, -1.8244363832497823, -1.6854436452303931,
        -1.553065810798341, -1.4665315654591369, -1.424383092073971,
    ]

    /// assert_allclose defaults, rtol 1e-7 and atol 0 (abTEM states no bound for this call).
    static func close(_ x: Double, _ y: Double) -> Bool { abs(x - y) <= 1e-7 * abs(y) }

    @Test("the relative difference matches abTEM")
    func differenceMatchesAbTEM() throws {
        let d = try Self.a.relativeDifference(to: Self.b)
        #expect(d.nx == 7 && d.ny == 9 && d.samplingX == 0.3 && d.samplingY == 0.25)
        #expect(d.values.count == Self.difference.count)
        for (x, y) in zip(d.values, Self.difference) { #expect(Self.close(x, y)) }
    }

    @Test("below the tolerance the difference is NaN, at the pixels abTEM marks")
    func differenceTolerance() throws {
        var v = Self.a.values
        v[1 * 9 + 1] = 0
        v[2 * 9 + 2] = -0.5
        let z = try RealField(nx: 7, ny: 9, values: v, samplingX: 0.3, samplingY: 0.25)
        let d = try z.relativeDifference(to: Self.b, minimumRelativeTolerance: 0.2)
        for i in 0..<63 {
            if i == 10 || i == 20 { #expect(d.values[i].isNaN) } else { #expect(Self.close(d.values[i], Self.difference[i])) }
        }
        // With the default 0 a zero pixel divides by zero, as NumPy's does.
        let loose = try z.relativeDifference(to: Self.b)
        #expect(loose.values[10] == -.infinity)
        #expect(loose.values[20] == ((-0.5 - Self.b.values[20]) / -0.5) * 100)
    }

    @Test("a NaN in the image makes every pixel NaN, as NumPy's max does")
    func differenceNaN() throws {
        var v = Self.a.values
        v[5] = .nan
        let n = try RealField(nx: 7, ny: 9, values: v, samplingX: 0.3, samplingY: 0.25)
        #expect(try n.relativeDifference(to: Self.b, minimumRelativeTolerance: 0.1).values.allSatisfy(\.isNaN))
    }

    @Test("an image is its own difference of zero")
    func differenceToItself() throws {
        #expect(try Self.a.relativeDifference(to: Self.a).values.allSatisfy { $0 == 0 })
    }

    @Test("images of another shape or another sampling are refused; a NaN tolerance gives NaN everywhere, as abTEM's")
    func differenceRefusals() throws {
        let other = try RealField(nx: 7, ny: 9, values: Self.a.values, samplingX: 0.3, samplingY: 0.2)
        #expect(throws: ImageOperationError.imagesDiffer) { try Self.a.relativeDifference(to: other) }
        let small = try Self.a.cropped(extentX: 1.8, extentY: 2.25)
        #expect(throws: ImageOperationError.imagesDiffer) { try Self.a.relativeDifference(to: small) }
        #expect(try Self.a.relativeDifference(to: Self.b, minimumRelativeTolerance: .nan).values.allSatisfy(\.isNaN), "as abTEM's")
    }
}

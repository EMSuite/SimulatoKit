//
//  ImageComparison.swift
//  Role: what a simulated image needs to sit beside a measured one: abTEM's `Images.crop` (a window of the image in Å) and
//        `BaseMeasurements.relative_difference` (the per-pixel difference in percent of the image). CPU float64; rows and
//        columns as everywhere in the kit: x is the first index, arrays are row-major `[ix * ny + iy]`.
//

import Foundation

extension RealField {
    /// The image's extent [Å] along x and y: points times sampling.
    public var extent: (x: Double, y: Double) { (Double(nx) * samplingX, Double(ny) * samplingY) }

    /// The image inside a window (abTEM `Images.crop(extent, offset)`): the window's lower corner and its size in pixels are
    /// `round(n × offset / image extent)` and `round(n × extent / image extent)`, rounded half to even as NumPy's `round`, and
    /// a window that runs past the far edge is cut there, as NumPy's slice is. The sampling stays.
    /// - Throws: `ImageOperationError.invalidCrop` for an extent or offset that is not finite, an extent that is not positive
    ///   or is larger than the image's (abTEM refuses that too), a negative offset, or a window that holds no pixel (abTEM
    ///   returns an empty image).
    public func cropped(extentX: Double, extentY: Double, offsetX: Double = 0, offsetY: Double = 0) throws -> RealField {
        guard extentX.isFinite, extentY.isFinite, offsetX.isFinite, offsetY.isFinite else {
            throw ImageOperationError.invalidCrop("the extent and offset must be finite, got (\(extentX), \(extentY)) Å at (\(offsetX), \(offsetY)) Å")
        }
        let full = extent
        guard extentX > 0, extentY > 0, extentX <= full.x, extentY <= full.y else {
            throw ImageOperationError.invalidCrop("the extent must be positive and at most the image's (\(full.x), \(full.y)) Å, got (\(extentX), \(extentY)) Å")
        }
        guard offsetX >= 0, offsetY >= 0 else {
            throw ImageOperationError.invalidCrop("the offset must be at least 0 Å, got (\(offsetX), \(offsetY)) Å")
        }
        // Both ratios are at most n (extent) or finite and nonnegative (offset); the offset is bounded before Int() below.
        let ox = Double(nx) * offsetX / full.x, oy = Double(ny) * offsetY / full.y
        guard ox < Double(nx), oy < Double(ny) else {
            throw ImageOperationError.invalidCrop("the window at (\(offsetX), \(offsetY)) Å starts outside the image")
        }
        let x0 = Int(ox.rounded(.toNearestOrEven)), y0 = Int(oy.rounded(.toNearestOrEven))
        let mx = Int((Double(nx) * extentX / full.x).rounded(.toNearestOrEven))
        let my = Int((Double(ny) * extentY / full.y).rounded(.toNearestOrEven))
        let x1 = Swift.min(x0 + mx, nx), y1 = Swift.min(y0 + my, ny)
        guard x1 > x0, y1 > y0 else {
            throw ImageOperationError.invalidCrop("the window of (\(extentX), \(extentY)) Å at (\(offsetX), \(offsetY)) Å holds no pixel")
        }
        let cy = y1 - y0
        var out = [Double]()
        out.reserveCapacity((x1 - x0) * cy)
        for ix in x0..<x1 { out.append(contentsOf: values[(ix * ny + y0)..<(ix * ny + y1)]) }
        return RealField.unchecked(nx: x1 - x0, ny: cy, values: out, samplingX: samplingX, samplingY: samplingY)
    }

    /// The image inside a window centred on the image (abTEM `Images.crop(extent, centered=True)`): the offset is half the
    /// image's extent minus half the window's, then as `cropped(extentX:extentY:offsetX:offsetY:)`.
    public func croppedCentered(extentX: Double, extentY: Double) throws -> RealField {
        let full = extent
        return try cropped(extentX: extentX, extentY: extentY, offsetX: full.x / 2 - extentX / 2, offsetY: full.y / 2 - extentY / 2)
    }

    /// The relative difference to another image in percent (abTEM `BaseMeasurements.relative_difference(other,
    /// min_relative_tol)`): 100 (a − b) / a where |a| ≥ `minimumRelativeTolerance` × max(a), NaN elsewhere, with a this image
    /// and b `other`, in abTEM's order (a − b, then / a, then × 100). With the default 0 a pixel where a is 0 divides by 0
    /// (± infinity, or NaN where b is 0 too), as abTEM's does. The sampling stays.
    /// One named departure: abTEM compares any two images of the same shape, whatever their samplings; this refuses images
    /// whose samplings differ, since their pixels are not the same places (resample first, `interpolated(samplingX:samplingY:)`).
    /// A NaN or infinite tolerance makes every pixel NaN, as abTEM's does.
    /// - Throws: `ImageOperationError.imagesDiffer` when the two differ in shape or sampling.
    public func relativeDifference(to other: RealField, minimumRelativeTolerance: Double = 0) throws -> RealField {
        guard nx == other.nx, ny == other.ny, samplingX == other.samplingX, samplingY == other.samplingY else {
            throw ImageOperationError.imagesDiffer
        }
        // NumPy's max propagates NaN: one NaN makes every comparison false, so every pixel is NaN.
        let peak = values.contains { $0.isNaN } ? Double.nan : (values.max() ?? 0)
        let floor = minimumRelativeTolerance * peak
        var out = [Double](repeating: 0, count: values.count)
        for i in values.indices {
            let a = values[i]
            out[i] = abs(a) >= floor ? ((a - other.values[i]) / a) * 100 : .nan
        }
        return RealField.unchecked(nx: nx, ny: ny, values: out, samplingX: samplingX, samplingY: samplingY)
    }
}

//
//  FlexibleAnnularPinsTests.swift
//  Role: `FlexibleAnnularDetector` against abTEM 1.0.10 (tools/pins/flexible-annular/pins.py): a 20 mrad probe at
//        200 kV scanned over 4 x 4 positions of Si [110] (M1) and SrTiO3 [001] (M2, M2_offset), 1 mrad bins. abTEM
//        returns the bins as float32, so they are held at 3e-7 of the largest bin (the float32 rounding bound of the
//        generator's own check); the AnnularDetector of the same limits is float64 and held at 1e-10 (the STEM pin).
//        The bins summed between two edges equal the kit's AnnularDetector; Metal holds the STEM detector tolerance.
//

import CryptoKit
import Foundation
import Testing
import SimulatoCore
import SimulatoDetector
import SimulatoMultislice
import SimulatoProbe

enum FlexiblePins {
    static let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Fixtures")

    struct Loaded: @unchecked Sendable {  // immutable after load
        let json: [String: Any]
        let blob: Data
    }

    static let loaded: Loaded = {
        let json = try! JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent("flexible-annular-pins.json"))) as! [String: Any]
        let arrays = json["arrays"] as! [String: Any]
        let blob = try! Data(contentsOf: directory.appendingPathComponent(arrays["file"] as! String))
        let digest = SHA256.hash(data: blob).map { String(format: "%02x", $0) }.joined()
        precondition(digest == arrays["sha256"] as! String, "flexible-annular-arrays.bin does not match its sha256")
        return Loaded(json: json, blob: blob)
    }()

    static func doubles(_ name: String) -> [Double] {
        let index = (loaded.json["arrays"] as! [String: Any])["index"] as! [String: Any]
        guard let entry = index[name] as? [String: Any] else { fatalError("no pinned array \(name)") }
        let offset = entry["offset"] as! Int, count = entry["count"] as! Int
        return loaded.blob.withUnsafeBytes { raw in
            (0..<count).map { i in
                Double(bitPattern: UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: offset + 8 * i, as: UInt64.self)))
            }
        }
    }

    static func `case`(_ id: String) -> [String: Any] { (loaded.json["cases"] as! [String: Any])[id] as! [String: Any] }

    struct Run {
        let detector: FlexibleAnnularDetector
        let segmented: SegmentedDetector
        let bins: ScanMeasurement
        let annular: ScanMeasurement
        /// The kit's AnnularDetector from 10 mrad above inner to 10 mrad below outer (bin edges).
        let band: ScanMeasurement
    }

    /// The case's scan through its static potential with abTEM's float32 π (the exit-wave pin's choice).
    static func run(_ id: String, backend: Backend = .cpu) throws -> Run {
        let c = Self.case(id)
        let box = c["box"] as! [Double], gpts = c["gpts"] as! [Int], scanGpts = c["scan_gpts"] as! [Int]
        let atoms = zip(c["numbers"] as! [Int], c["positions"] as! [[Double]]).map {
            AtomicStructure.Atom(atomicNumber: $0.0, position: SIMD3($0.1[0], $0.1[1], $0.1[2]))
        }
        let structure = try AtomicStructure(box: SIMD3(box[0], box[1], box[2]), atoms: atoms)
        let potential = try ProjectedPotential(structure: structure, nx: gpts[0], ny: gpts[1],
                                               sliceThickness: c["slice_thickness"] as! Double,
                                               parameterization: .lobato(LobatoParameterization(piConstant: .abTEMFloat32)))
        let probe = try Probe(beam: ElectronBeam(energy: c["energy"] as! Double), grid: potential.grid,
                              aperture: Aperture(semiangleCutoff: c["semiangle_cutoff_mrad"] as! Double))
        let scan = try GridScan(start: SIMD2(0, 0), end: SIMD2(box[0], box[1]), nx: scanGpts[0], ny: scanGpts[1])
        let inner = c["inner_mrad"] as! Double, outer = c["outer_mrad"] as! Double
        let detector = try FlexibleAnnularDetector(step: c["step_mrad"] as! Double, inner: inner, outer: outer)
        let detectors: [any Detector] = [detector, try AnnularDetector(inner: inner, outer: outer),
                                         try AnnularDetector(inner: inner + 10, outer: outer - 10)]
        let m = try STEM.scan(probe: probe, potentials: [potential], scan: scan, detectors: detectors, backend: backend)
        return Run(detector: detector, segmented: try detector.segmented(for: probe.grid, beam: probe.beam),
                   bins: m[0], annular: m[1], band: m[2])
    }
}

@Suite("Flexible annular detector against abTEM")
struct FlexibleAnnularPinsTests {
    static let cases = ["M1", "M2", "M2_offset"]

    @Test(arguments: cases)
    func binsEqualAbTEMs(id: String) throws {
        let r = try FlexiblePins.run(id)
        let nbins = FlexiblePins.case(id)["nbins"] as! Int
        #expect(r.bins.valueShape == [nbins, 1])
        #expect(r.segmented.radialBins == nbins)
        let pinned = FlexiblePins.doubles("\(id).bins")
        #expect(r.bins.values.count == pinned.count)
        let scale = pinned.map(abs).max()!
        let worst = zip(r.bins.values, pinned).map { abs($0 - $1) }.max()! / scale
        #expect(worst <= 3e-7, "\(id): bins within \(worst) of the largest")

        let annular = FlexiblePins.doubles("\(id).annular")
        let annularScale = annular.map(abs).max()!
        let worstAnnular = zip(r.annular.values, annular).map { abs($0 - $1) }.max()! / annularScale
        #expect(worstAnnular <= 1e-10, "\(id): AnnularDetector within \(worstAnnular)")
    }

    /// The bins summed between two of their edges equal the AnnularDetector of those limits at every position, the
    /// whole range and a band inside it (the app's presets are such bands).
    @Test(arguments: cases)
    func binsIntegrateToTheAnnularDetector(id: String) throws {
        let r = try FlexiblePins.run(id)
        let whole = try r.bins.integrateRadial(inner: r.segmented.inner, outer: r.segmented.outer, detector: r.segmented)
        let scale = r.annular.values.map(abs).max()!
        #expect(zip(whole.values, r.annular.values).allSatisfy { abs($0 - $1) <= 1e-12 * scale }, "\(id) whole range")

        let banded = try r.bins.integrateRadial(inner: r.segmented.inner + 10, outer: r.segmented.outer - 10,
                                                detector: r.segmented)
        let bandScale = r.band.values.map(abs).max()!
        let worst = zip(banded.values, r.band.values).map { abs($0 - $1) }.max()! / bandScale
        #expect(worst <= 1e-12, "\(id) band: \(worst)")
    }

    @Test(.enabled(if: Multislice.metalIsAvailable, "no Metal device: the Metal flexible detector is not tested here"),
          arguments: cases)
    func metalBinsMatchTheCPUBins(id: String) throws {
        let cpu = try FlexiblePins.run(id).bins.values
        let gpu = try FlexiblePins.run(id, backend: .metal).bins.values
        let scale = cpu.map(abs).max()!
        let d = zip(gpu, cpu).map { abs($0 - $1) }.max()! / scale
        print("Metal vs CPU flexible bins, \(id): \(d)")
        #expect(d <= 2e-4, "\(id): Metal against CPU \(d)")
    }

    @Test func refusals() throws {
        #expect(throws: SimulatoError.self) { try FlexibleAnnularDetector(step: 0) }
        #expect(throws: SimulatoError.self) { try FlexibleAnnularDetector(inner: -1) }
        #expect(throws: SimulatoError.self) { try FlexibleAnnularDetector(inner: 50, outer: 50) }
        // Beyond the pattern's maximum angle.
        let grid = try Grid(nx: 64, ny: 64, extentX: 6.4, extentY: 6.4)
        let beam = try ElectronBeam(energy: 200e3)
        #expect(throws: SimulatoError.self) {
            try FlexibleAnnularDetector(outer: 400).segmented(for: grid, beam: beam)
        }
    }
}

@Suite("Flexible annular detector defaults")
struct FlexibleAnnularDefaultTests {
    /// abTEM's polar detectors take min(cutoff angles) for a missing outer angle, unrounded (`_match_waves`).
    @Test func missingOuterIsTheCutoffAngle() throws {
        let grid = try Grid(nx: 64, ny: 64, extentX: 6.4, extentY: 6.4)
        let beam = try ElectronBeam(energy: 200e3)
        let c = Antialias.cutoffAngles(on: grid, wavelength: beam.wavelength)
        let s = try FlexibleAnnularDetector().segmented(for: grid, beam: beam)
        #expect(s.outer == min(c.x, c.y))
        #expect(s.radialBins == Int(floor(min(c.x, c.y))))
    }
}

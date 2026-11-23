//
//  SlabPinsTests.swift
//  Role: the slab builder against abTEM 1.0.10 (tools/pins/slab/pins.py): the orthogonal unit of every named structure and
//        zone axis (box, scale and shear at rtol 1e-7, assert_allclose's float64 default; supercell matrix and atom count
//        exact; atoms as periodic sets at rtol 1e-7, atol 1e-9 Å), the repetition ladder and its refusal, the orientation
//        against py4DSTEM's plain matrices, and the exit wave of the built Si [110] slab at the kit's multislice pin.
//

import SimulatoKitTestSupport
import CryptoKit
import Foundation
import simd
import Testing
import SimulatoCore
import SimulatoMultislice

enum SlabPins {
    static let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Fixtures")

    struct Loaded: @unchecked Sendable {  // immutable after load
        let json: [String: Any]
        let blob: Data
    }

    static let loaded: Loaded = {
        let json = try! JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent("slab-pins.json"))) as! [String: Any]
        let arrays = json["arrays"] as! [String: Any]
        let blob = try! Data(contentsOf: directory.appendingPathComponent(arrays["file"] as! String))
        let digest = SHA256.hash(data: blob).map { String(format: "%02x", $0) }.joined()
        precondition(digest == arrays["sha256"] as! String, "slab-arrays.bin does not match its sha256 in the pins")
        return Loaded(json: json, blob: blob)
    }()

    static var json: [String: Any] { loaded.json }

    static func complex(_ name: String) -> ComplexField {
        let index = (json["arrays"] as! [String: Any])["index"] as! [String: Any]
        let entry = index[name] as! [String: Any]
        let offset = entry["offset"] as! Int, count = entry["count"] as! Int, shape = entry["shape"] as! [Int]
        let d: [Double] = loaded.blob.withUnsafeBytes { raw in
            (0..<(2 * count)).map { i in
                Double(bitPattern: UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: offset + 8 * i, as: UInt64.self)))
            }
        }
        return ComplexField(nx: shape[0], ny: shape[1], real: stride(from: 0, to: d.count, by: 2).map { d[$0] },
                            imaginary: stride(from: 1, to: d.count, by: 2).map { d[$0] })
    }

    static func vector(_ a: Any) -> SIMD3<Double> {
        let v = a as! [Double]
        return SIMD3(v[0], v[1], v[2])
    }

    static func cell(_ name: String) throws -> CrystalCell {
        let s = (json["structures"] as! [String: Any])[name] as! [String: Any]
        let rows = (s["cell"] as! [[Double]]).map { SIMD3($0[0], $0[1], $0[2]) }
        let sites = (s["sites"] as! [[Double]]).map {
            CrystalCell.Site(atomicNumber: Int($0[0]), fractional: SIMD3($0[1], $0[2], $0[3]))
        }
        return try CrystalCell(a: rows[0], b: rows[1], c: rows[2], sites: sites)
    }

    static func zone(_ a: Any) -> SIMD3<Int> {
        let z = a as! [Int]
        return SIMD3(z[0], z[1], z[2])
    }

    static var units: [[String: Any]] { json["units"] as! [[String: Any]] }
    static var ladders: [[String: Any]] { json["ladders"] as! [[String: Any]] }
    static var orientations: [[String: Any]] { json["orientation"] as! [[String: Any]] }
    static var slab: [String: Any] { json["slab"] as! [String: Any] }
}

/// The worst distance between two sets of atoms in a periodic box, matching each atom to the nearest one of the same Z (no
/// atom used twice), relative to `atol + rtol · |position|`; infinity when the counts or the elements differ.
private func periodicSetMismatch(_ got: [AtomicStructure.Atom], _ want: [(Int, SIMD3<Double>)], box: SIMD3<Double>,
                         rtol: Double = 1e-7, atol: Double = 1e-9) -> Double {
    guard got.count == want.count else { return .infinity }
    var used = [Bool](repeating: false, count: want.count)
    var worst = 0.0
    for atom in got {
        var best = -1, bestDistance = Double.infinity
        for (j, w) in want.enumerated() where !used[j] && w.0 == atom.atomicNumber {
            var d = atom.position - w.1
            for a in 0..<3 { d[a] -= box[a] * (d[a] / box[a]).rounded() }
            let distance = (d * d).sum().squareRoot()
            if distance < bestDistance {
                best = j
                bestDistance = distance
            }
        }
        guard best >= 0 else { return .infinity }
        used[best] = true
        let scale = atol + rtol * (want[best].1 * want[best].1).sum().squareRoot()
        worst = max(worst, bestDistance / scale)
    }
    return worst
}

/// assert_allclose's comparison: |a − b| ≤ atol + rtol |b|.
private func allclose(_ a: SIMD3<Double>, _ b: SIMD3<Double>, rtol: Double = 1e-7, atol: Double = 1e-12) -> Bool {
    (0..<3).allSatisfy { abs(a[$0] - b[$0]) <= atol + rtol * abs(b[$0]) }
}

private func columns(_ m: simd_double3x3) -> [SIMD3<Double>] { [m.columns.0, m.columns.1, m.columns.2] }

@Suite("Slab builder against abTEM")
struct SlabPinsTests {
    struct UnitCase: Sendable, CustomTestStringConvertible {
        static let all = SlabPins.units.indices.map(UnitCase.init)
        let index: Int
        var testDescription: String {
            let u = SlabPins.units[index]
            return "\(u["structure"] as! String) \(u["zone"] as! [Int]) n=\(u["repetitions"] as! Int)"
        }
    }

    struct LadderCase: Sendable, CustomTestStringConvertible {
        static let all = SlabPins.ladders.indices.map(LadderCase.init)
        let index: Int
        var testDescription: String {
            let l = SlabPins.ladders[index]
            return "\(l["structure"] as! String) \(l["zone"] as! [Int])"
        }
    }

    @Test(arguments: UnitCase.all)
    func orthogonalUnitEqualsAbTEM(c: UnitCase) throws {
        let pin = SlabPins.units[c.index]
        let tag = c.testDescription
        let cell = try SlabPins.cell(pin["structure"] as! String)
        let unit = try SlabBuilder.orthogonalUnit(of: cell, zoneAxis: SlabPins.zone(pin["zone"]!),
                                                  maximumRepetitions: pin["repetitions"] as! Int)
        let box = SlabPins.vector(pin["box"]!)
        #expect(allclose(unit.structure.box, box), "\(tag) box \(unit.structure.box) vs \(box)")
        let supercell = (pin["supercell"] as! [[Int]]).map { SIMD3($0[0], $0[1], $0[2]) }
        #expect(unit.supercell == supercell, "\(tag) supercell \(unit.supercell) vs \(supercell)")
        let numbers = pin["numbers"] as! [Int]
        #expect(unit.structure.atoms.count == numbers.count, "\(tag) atom count")
        #expect(unit.structure.atoms.count == unit.cellCount * cell.sites.count, "\(tag) det V cells")
        let want = zip(numbers, pin["positions"] as! [[Double]]).map { ($0.0, SIMD3($0.1[0], $0.1[1], $0.1[2])) }
        let mismatch = periodicSetMismatch(unit.structure.atoms, want, box: box)
        #expect(mismatch <= 1, "\(tag) atoms: worst \(mismatch) of the bound")
        #expect(allclose(unit.scale, SlabPins.vector(pin["scale"]!)), "\(tag) scale \(unit.scale)")
        #expect(allclose(unit.shear, SlabPins.vector(pin["shear"]!)), "\(tag) shear \(unit.shear)")
        #expect(abs(unit.strain - (pin["strain"] as! Double)) <= 1e-12 + 1e-7 * (pin["strain"] as! Double), "\(tag) strain")
        // Every coordinate lies in [0, side), none within 1e-6 Å below the far face.
        for atom in unit.structure.atoms {
            for a in 0..<3 {
                #expect(atom.position[a] >= 0 && atom.position[a] <= unit.structure.box[a] - 1e-6, "\(tag) atom in the box")
            }
        }
    }

    @Test(arguments: LadderCase.all)
    func ladderStopsWhereAbTEMHoldsTheStrain(c: LadderCase) throws {
        let pin = SlabPins.ladders[c.index]
        let tag = c.testDescription
        let cell = try SlabPins.cell(pin["structure"] as! String)
        let zone = SlabPins.zone(pin["zone"]!)
        let orientation = (pin["orientation"] as! [[Double]]).map { SIMD3($0[0], $0[1], $0[2]) }  // rows of the matrix
        if let refused = pin["refused"] as? [String: Any] {
            do {
                _ = try SlabBuilder.orthogonalUnit(of: cell, zoneAxis: zone)
                Issue.record("\(tag) was not refused")
            } catch let SimulatoError.strainTooLarge(strain, maximum, repetitions) {
                let want = refused["strain"] as! Double
                #expect(abs(strain - want) <= 1e-7 * want, "\(tag) refused with strain \(strain), abTEM \(want)")
                #expect(maximum == SlabBuilder.defaultMaximumStrain && strain > maximum, "\(tag) limit")
                #expect(repetitions == refused["repetitions"] as! Int, "\(tag) repetitions \(repetitions)")
            }
            // The refusal is a choice: a looser limit accepts the unit.
            let loose = try SlabBuilder.orthogonalUnit(of: cell, zoneAxis: zone, maximumStrain: 0.01)
            #expect(loose.strain <= 0.01, "\(tag) accepted at 1 %")
        } else {
            let unit = try SlabBuilder.orthogonalUnit(of: cell, zoneAxis: zone)
            #expect(unit.maximumRepetitions == pin["repetitions"] as! Int, "\(tag) stops at \(unit.maximumRepetitions)")
            #expect(unit.strain <= SlabBuilder.defaultMaximumStrain, "\(tag) strain \(unit.strain)")
            #expect(allclose(unit.structure.box, SlabPins.vector(pin["box"]!)), "\(tag) box")
            // The orientation (the kit's own rule, restated by the generator): rows of the pinned matrix are its columns.
            let got = columns(unit.orientation)
            for i in 0..<3 {
                #expect(simd_length(SIMD3(got[0][i], got[1][i], got[2][i]) - orientation[i]) <= 1e-12, "\(tag) orientation row \(i)")
            }
        }
    }

    /// The suite's convention, checked on py4DSTEM's plain matrices (DiffractoKit's crystal-orientation pins): a proper
    /// rotation, the third column the zone axis pointing at the source, and a turn about the beam only away from
    /// py4DSTEM's in-plane reference (the kit puts a lattice row along x). On [0 0 1] of a cell with a along x the two are
    /// the same matrix.
    @Test func orientationFollowsTheSuiteConvention() throws {
        for pin in SlabPins.orientations {
            let rows = (pin["cell"] as! [[Double]]).map { SIMD3($0[0], $0[1], $0[2]) }
            let cell = try CrystalCell(a: rows[0], b: rows[1], c: rows[2],
                                       sites: [CrystalCell.Site(atomicNumber: 13, fractional: .zero)])
            let zone = SlabPins.zone(pin["zone"]!)
            let tag = "\(pin["phase"] as! String) \(pin["zone"] as! [Int])"
            let m = pin["matrix"] as! [Double]  // row-major
            let py4DSTEM = simd_double3x3(rows: [SIMD3(m[0], m[1], m[2]), SIMD3(m[3], m[4], m[5]), SIMD3(m[6], m[7], m[8])])
            let got = try SlabBuilder.orthogonalUnit(of: cell, zoneAxis: zone, maximumRepetitions: 5).orientation
            let identity = got.transpose * got
            for (i, c) in columns(identity).enumerated() {
                var e = SIMD3<Double>.zero
                e[i] = 1
                #expect(simd_length(c - e) <= 1e-12, "\(tag) orthonormal")
            }
            #expect(abs(simd_determinant(got) - 1) <= 1e-12, "\(tag) proper")
            #expect(simd_length(got.columns.2 - py4DSTEM.columns.2) <= 1e-12, "\(tag) third column is the zone axis")
            let turn = py4DSTEM.transpose * got  // a rotation about z
            #expect(abs(turn[2][2] - 1) <= 1e-12 && abs(turn[0][2]) <= 1e-12 && abs(turn[1][2]) <= 1e-12,
                    "\(tag) a turn about the beam only")
            if zone == SIMD3(0, 0, 1), ["Al", "Mg"].contains(pin["phase"] as! String) {
                for (a, b) in zip(columns(got), columns(py4DSTEM)) {
                    #expect(simd_length(a - b) <= 1e-12, "\(tag) equals py4DSTEM's matrix")
                }
            }
        }
    }

    func builtSlab() throws -> Slab {
        let pin = SlabPins.slab
        let cell = try SlabPins.cell(pin["structure"] as! String)
        let unit = try SlabBuilder.orthogonalUnit(of: cell, zoneAxis: SlabPins.zone(pin["zone"]!))
        return try SlabBuilder.slab(of: cell, zoneAxis: SlabPins.zone(pin["zone"]!),
                                    thickness: unit.structure.box.z * Double(pin["periods"] as! Int),
                                    sliceThickness: pin["slice_thickness"] as! Double)
    }

    @Test func siliconSlabEqualsTheGeneratorsAndKeepsPlanesOffSliceBoundaries() throws {
        let pin = SlabPins.slab
        let slab = try builtSlab()
        let box = SlabPins.vector(pin["box"]!)
        #expect(allclose(slab.structure.box, box), "box \(slab.structure.box)")
        #expect(slab.repeats == SIMD3(1, 1, pin["periods"] as! Int))
        #expect(slab.sliceThicknesses.count == pin["slices"] as! Int, "slices \(slab.sliceThicknesses.count)")
        let h = pin["slice_height"] as! Double
        #expect(slab.sliceThicknesses.allSatisfy { abs($0 - h) <= 1e-12 * h }, "slice height")
        #expect(abs(slab.zShift - (pin["z_shift"] as! Double)) <= 1e-12, "z shift \(slab.zShift)")
        #expect(slab.planeMargin >= 0.4, "plane margin \(slab.planeMargin) Å")
        let want = zip(pin["numbers"] as! [Int], pin["positions"] as! [[Double]]).map { ($0.0, SIMD3($0.1[0], $0.1[1], $0.1[2])) }
        let mismatch = periodicSetMismatch(slab.structure.atoms, want, box: box)
        #expect(mismatch <= 1, "slab atoms: worst \(mismatch) of the bound")
        // No atom within 0.4 Å of a slice boundary (truth, any reference aside).
        for atom in slab.structure.atoms {
            let q = atom.position.z / h
            #expect(abs(q - q.rounded()) * h >= 0.4, "atom at z = \(atom.position.z) Å")
        }
    }

    @Test(arguments: [(LobatoParameterization.PiConstant.abTEMFloat32, 1e-10), (.exact, 1e-6)])
    func siliconSlabExitWaveEqualsAbTEM(pi: LobatoParameterization.PiConstant, bound: Double) throws {
        let pin = SlabPins.slab
        let slab = try builtSlab()
        let gpts = pin["gpts"] as! [Int]
        let potential = try Potential(structure: slab.structure, nx: gpts[0], ny: gpts[1],
                                      sliceThicknesses: slab.sliceThicknesses,
                                      parameterization: .lobato(LobatoParameterization(piConstant: pi)))
        let beam = try ElectronBeam(energy: pin["energy"] as! Double)
        let waves = try Multislice.run(Waves.planeWave(beam: beam, grid: potential.grid), through: potential)
        let reference = SlabPins.complex("si110.exit")
        #expect(relMax(waves[0].field, reference) <= bound, "Si [110] exit wave: \(relMax(waves[0].field, reference))")
    }
}

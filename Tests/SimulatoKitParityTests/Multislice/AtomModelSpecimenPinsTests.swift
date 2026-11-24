//
//  AtomModelSpecimenPinsTests.swift
//  Role: specimens from atom models against abTEM 1.0.10 and ASE 3.29.0 (tools/pins/atom-models/pins.py): Al with a
//        vacancy and a Σ5 (310) bicrystal pass through unchanged (rtol 1e-7, atol 1e-9 Å) and their exit waves equal
//        abTEM's at the kit's pin (1e-10 of the maximum with abTEM's float32 π); an orthorhombic Al model with the beam
//        along x, y and z equals `orthogonalize_cell(plane=)` (box, atoms as sets, exit wave); ASE's `center(vacuum=)`
//        on a model without a cell and on two and one axes of a rectangular cell (rtol 1e-12).
//

import CryptoKit
import Foundation
import Testing
import SimulatoCore
import SimulatoKitTestSupport
import SimulatoMultislice

enum AtomModelPins {
    static let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Fixtures")

    struct Loaded: @unchecked Sendable {  // immutable after load
        let json: [String: Any]
        let blob: Data
    }

    static let loaded: Loaded = {
        let json = try! JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent("atom-models-pins.json"))) as! [String: Any]
        let arrays = json["arrays"] as! [String: Any]
        let blob = try! Data(contentsOf: directory.appendingPathComponent(arrays["file"] as! String))
        let digest = SHA256.hash(data: blob).map { String(format: "%02x", $0) }.joined()
        precondition(digest == arrays["sha256"] as! String, "atom-models-arrays.bin does not match its sha256")
        return Loaded(json: json, blob: blob)
    }()

    static var cases: [String: Any] { loaded.json["cases"] as! [String: Any] }

    static func exitWave(_ name: String) -> ComplexField {
        let index = (loaded.json["arrays"] as! [String: Any])["index"] as! [String: Any]
        guard let entry = index["\(name).exit_wave"] as? [String: Any] else { fatalError("no pinned exit wave \(name)") }
        let offset = entry["offset"] as! Int, count = entry["count"] as! Int, shape = entry["shape"] as! [Int]
        let d: [Double] = loaded.blob.withUnsafeBytes { raw in
            (0..<(2 * count)).map { i in
                Double(bitPattern: UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: offset + 8 * i, as: UInt64.self)))
            }
        }
        return ComplexField.unchecked(nx: shape[0], ny: shape[1], real: stride(from: 0, to: d.count, by: 2).map { d[$0] },
                                      imaginary: stride(from: 1, to: d.count, by: 2).map { d[$0] })
    }

    static func vector(_ v: Any) -> SIMD3<Double> {
        let a = v as! [Double]
        return SIMD3(a[0], a[1], a[2])
    }

    /// A pinned model: its rectangular box, atomic numbers and positions.
    static func model(_ entry: [String: Any]) -> (box: SIMD3<Double>, numbers: [Int], positions: [SIMD3<Double>]) {
        (vector(entry["box"]!), entry["numbers"] as! [Int], (entry["positions"] as! [[Double]]).map(vector))
    }

    /// The exit wave of an orthogonal unit, the plane wave through the pin's slices.
    static func exitWave(of unit: OrthogonalUnit, gpts: [Int]) throws -> ComplexField {
        let potential = try Potential(structure: unit.structure, nx: gpts[0], ny: gpts[1],
                                      sliceThickness: loaded.json["slice_thickness"] as! Double,
                                      parameterization: .lobato(LobatoParameterization(piConstant: .abTEMFloat32)))
        let beam = try ElectronBeam(energy: loaded.json["energy"] as! Double)
        return try Multislice.run(Waves.planeWave(beam: beam, grid: potential.grid), through: potential)[0].field
    }

    /// Every pinned atom has one of the same Z in `atoms` within `tolerance` Å, and the counts agree.
    static func sameAtoms(_ atoms: [AtomicStructure.Atom], numbers: [Int], positions: [SIMD3<Double>],
                          tolerance: Double) -> Bool {
        guard atoms.count == numbers.count else { return false }
        var used = [Bool](repeating: false, count: atoms.count)
        for (z, p) in zip(numbers, positions) {
            guard let j = atoms.indices.first(where: { k in
                let d = atoms[k].position - p
                return !used[k] && atoms[k].atomicNumber == z && (d * d).sum() <= tolerance * tolerance
            }) else { return false }
            used[j] = true
        }
        return true
    }
}

@Suite("Specimens from atom models against abTEM and ASE")
struct AtomModelSpecimenPinsTests {
    /// An orthogonal model passes through unchanged (positions in order, apart from rounding), and its exit wave is abTEM's.
    @Test(arguments: ["vacancy", "sigma5"])
    func orthogonalModelPassesThroughAndItsExitWaveIsAbTEMs(id: String) throws {
        let entry = AtomModelPins.cases[id] as! [String: Any]
        let m = AtomModelPins.model(entry)
        let rows = [SIMD3(m.box.x, 0, 0), SIMD3(0, m.box.y, 0), SIMD3(0, 0, m.box.z)]
        let model = try CrystalCell(cellRows: rows, atomicNumbers: m.numbers, positions: m.positions)
        let unit = try SlabBuilder.orthogonalUnit(ofModel: model)
        #expect(unit.structure.box == m.box)
        #expect(unit.strain == 0 && unit.cellCount == 1)
        #expect(unit.structure.atoms.map(\.atomicNumber) == m.numbers)
        for (a, p) in zip(unit.structure.atoms, m.positions) {
            let d = a.position - p
            #expect(abs(d.x) <= 1e-9 + 1e-7 * abs(p.x) && abs(d.y) <= 1e-9 + 1e-7 * abs(p.y)
                        && abs(d.z) <= 1e-9 + 1e-7 * abs(p.z), "\(id): \(a.position) against \(p)")
        }
        let wave = try AtomModelPins.exitWave(of: unit, gpts: entry["gpts"] as! [Int])
        let reference = AtomModelPins.exitWave(id)
        #expect(relMax(wave, reference) <= 1e-10, "\(id) exit wave: \(relMax(wave, reference))")
        if id == "vacancy" { #expect(m.numbers.count == (entry["full_atoms_before_vacancy"] as! Int) - 1) }
    }

    /// The beam along x, y and z of an orthorhombic 2 x 3 x 4 Al model equals abTEM's planes "yz", "zx" and "xy".
    @Test(arguments: [("a", SlabBuilder.BeamAxis.x), ("b", .y), ("c", .z)])
    func beamAlongEachAxisEqualsAbTEMsPlane(beam: String, axis: SlabBuilder.BeamAxis) throws {
        let ortho = AtomModelPins.cases["ortho"] as! [String: Any]
        let pinned = (ortho["beams"] as! [String: Any])[beam] as! [String: Any]
        let cubic = try AtomicStructure(box: SIMD3(4.05, 4.05, 4.05), atoms: [
            SIMD3(0, 0, 0), SIMD3(0, 2.025, 2.025), SIMD3(2.025, 0, 2.025), SIMD3(2.025, 2.025, 0),
        ].map { AtomicStructure.Atom(atomicNumber: 13, position: $0) }).repeated(2, 3, 4)
        let box = AtomModelPins.vector(ortho["box"]!)
        let model = try CrystalCell(cellRows: [SIMD3(box.x, 0, 0), SIMD3(0, box.y, 0), SIMD3(0, 0, box.z)],
                                    atomicNumbers: cubic.atoms.map(\.atomicNumber), positions: cubic.atoms.map(\.position))
        let unit = try SlabBuilder.orthogonalUnit(ofModel: model, beamAlong: axis)
        let m = AtomModelPins.model(pinned)
        let side = unit.structure.box - m.box
        #expect(abs(side.x) <= 1e-7 * m.box.x && abs(side.y) <= 1e-7 * m.box.y && abs(side.z) <= 1e-7 * m.box.z,
                "beam \(beam): box \(unit.structure.box) against \(m.box)")
        #expect(unit.strain <= 1e-12, "beam \(beam): strain \(unit.strain)")
        #expect(AtomModelPins.sameAtoms(unit.structure.atoms, numbers: m.numbers, positions: m.positions, tolerance: 1e-6),
                "beam \(beam): the atoms differ")
        let wave = try AtomModelPins.exitWave(of: unit, gpts: pinned["gpts"] as! [Int])
        let reference = AtomModelPins.exitWave("ortho_\(beam)")
        #expect(relMax(wave, reference) <= 1e-10, "beam \(beam) exit wave: \(relMax(wave, reference))")
    }

    /// ASE's `center(vacuum=)`: a model without a cell on every axis; a 6 x 7 x 8 Å cell on x and y, and on z only.
    @Test(arguments: ["all", "axis_xy", "z_only"])
    func paddingEqualsASEsCenter(id: String) throws {
        let entry = (AtomModelPins.cases["padding"] as! [String: Any])[id] as! [String: Any]
        let arguments = entry["arguments"] as! [String: Any]
        let vacuum = arguments["vacuum"] as! Double
        let axes: Set<Int> = (arguments["axis"] as? [Int]).map(Set.init) ?? (arguments["axis"] as? Int).map { [$0] } ?? [0, 1, 2]
        let numbers = entry["numbers"] as! [Int]
        let pinned = (entry["positions"] as! [[Double]]).map(AtomModelPins.vector)
        let cell = (entry["cell"] as! [[Double]]).map(AtomModelPins.vector)
        // The model before centring: the pinned atoms moved along the padded axes only (centring forgets where they were).
        let offset = SIMD3<Double>(0.37, -0.21, 0.55)
        let start: [SIMD3<Double>] = pinned.map { p in
            var q = p
            for i in axes { q[i] += offset[i] }
            return q
        }
        let box: SIMD3<Double>? = id == "all" ? nil : SIMD3(6, 7, 8)
        let padded = try CrystalCell.padded(atomicNumbers: numbers, positions: start, box: box, vacuum: vacuum, axes: axes)
        for i in 0..<3 {
            let want = cell[i][i]
            let got = [padded.a, padded.b, padded.c][i][i]
            #expect(abs(got - want) <= 1e-12 * want, "\(id) side \(i): \(got) against \(want)")
        }
        // Positions: ASE's on every axis (padded: the nearest atom `vacuum` from the face; the others as given).
        let rows = [padded.a, padded.b, padded.c]
        for (site, p) in zip(padded.sites, pinned) {
            let q = site.fractional.x * rows[0] + site.fractional.y * rows[1] + site.fractional.z * rows[2]
            for i in 0..<3 {
                let want = p[i]
                #expect(abs(q[i] - want) <= 1e-12 * max(1, abs(want)), "\(id) axis \(i): \(q[i]) against \(want)")
            }
        }
    }

    @Test func refusals() throws {
        let p = [SIMD3<Double>(0, 0, 0)]
        #expect(throws: SimulatoError.self) {  // one atom, no vacuum: no extent
            try CrystalCell.padded(atomicNumbers: [13], positions: p, box: nil, vacuum: 0)
        }
        #expect(throws: SimulatoError.self) {  // no cell needs every axis padded
            try CrystalCell.padded(atomicNumbers: [13], positions: p, box: nil, vacuum: 3, axes: [2])
        }
        #expect(throws: SimulatoError.self) {
            try CrystalCell.padded(atomicNumbers: [13], positions: p, box: nil, vacuum: -1)
        }
        #expect(throws: SimulatoError.self) {
            try CrystalCell(cellRows: [SIMD3(4, 0, 0), SIMD3(0, 4, 0), SIMD3(0, 0, 4)], atomicNumbers: [13, 13], positions: p)
        }
        // A sheared cell beyond the strain limit names its strain.
        let sheared = try CrystalCell(cellRows: [SIMD3(4.05, 0, 0), SIMD3(1.3, 4.05, 0), SIMD3(0, 0, 4.05)],
                                      atomicNumbers: [13], positions: [.zero])
        let error = #expect(throws: SimulatoError.self) {
            try SlabBuilder.orthogonalUnit(ofModel: sheared, maximumStrain: 0)
        }
        if case .strainTooLarge(let strain, _, _)? = error {
            #expect(strain > 0)
        } else {
            Issue.record("expected strainTooLarge, got \(String(describing: error))")
        }
    }
}

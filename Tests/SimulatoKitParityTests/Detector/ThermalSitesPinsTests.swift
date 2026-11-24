//
//  ThermalSitesPinsTests.swift
//  Role: thermal vibration amplitude per site against abTEM 1.0.10 (tools/pins/thermal-sites/pins.py, #95): Si and
//        SrTiO₃ with σ per element (T1, T3) and per site (T2, T4). abTEM's displacements are given to the port (the
//        generator asserts they reproduce abTEM's displaced positions to the last bit), and every slice of every
//        configuration's potential equals abTEM's at the frozen-phonon pin (1e-11 of the largest value, abTEM's float32
//        π), as F1. σ per atom from σ per element equals abTEM's resolution exactly; σ ↔ B equal abTEM's `sigma_to_B`
//        and `B_to_sigma` at rtol 1e-7.
//

import CryptoKit
import Foundation
import Testing
import SimulatoCore
import SimulatoMultislice

enum ThermalPins {
    static let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Fixtures")

    struct Loaded: @unchecked Sendable {  // immutable after load
        let json: [String: Any]
        let blob: Data
    }

    static let loaded: Loaded = {
        let json = try! JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent("thermal-sites-pins.json"))) as! [String: Any]
        let arrays = json["arrays"] as! [String: Any]
        let blob = try! Data(contentsOf: directory.appendingPathComponent(arrays["file"] as! String))
        let digest = SHA256.hash(data: blob).map { String(format: "%02x", $0) }.joined()
        precondition(digest == arrays["sha256"] as! String, "thermal-sites-arrays.bin does not match its sha256 in the pins")
        return Loaded(json: json, blob: blob)
    }()

    static var json: [String: Any] { loaded.json }

    static func doubles(_ name: String) -> [Double] {
        let index = (json["arrays"] as! [String: Any])["index"] as! [String: Any]
        guard let entry = index[name] as? [String: Any] else { fatalError("no pinned array \(name)") }
        let offset = entry["offset"] as! Int
        let count = entry["count"] as! Int
        return loaded.blob.withUnsafeBytes { raw in
            (0..<count).map { i in
                Double(bitPattern: UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: offset + 8 * i, as: UInt64.self)))
            }
        }
    }

    static func `case`(_ id: String) -> [String: Any] { (json["cases"] as! [String: Any])[id] as! [String: Any] }

    static func structure(_ id: String) throws -> AtomicStructure {
        let c = Self.case(id)
        let box = c["box"] as! [Double]
        let numbers = c["numbers"] as! [Int]
        let positions = c["positions"] as! [[Double]]
        var atoms: [AtomicStructure.Atom] = []
        for (z, p) in zip(numbers, positions) {
            atoms.append(AtomicStructure.Atom(atomicNumber: z, position: SIMD3(p[0], p[1], p[2])))
        }
        return try AtomicStructure(box: SIMD3(box[0], box[1], box[2]), atoms: atoms)
    }

    /// abTEM's displacements of every configuration, given to the port.
    static func frozenPhonons(_ id: String) throws -> FrozenPhonons {
        let n = (Self.case(id)["seeds"] as! [Int]).count
        var all: [[SIMD3<Double>]] = []
        for c in 0..<n {
            let v = doubles("\(id).displacement.\(c)")
            var d: [SIMD3<Double>] = []
            for i in stride(from: 0, to: v.count, by: 3) { d.append(SIMD3(v[i], v[i + 1], v[i + 2])) }
            all.append(d)
        }
        return try FrozenPhonons(structure: structure(id), displacements: all)
    }

    /// σ per element of a T1/T3 case, by atomic number.
    static func elementSigmas(_ id: String) -> [Int: Double] {
        let bySymbol = Self.case(id)["sigmas"] as! [String: Double]
        let numbers = Self.case(id)["numbers"] as! [Int]
        let symbols: [Int: String] = [8: "O", 14: "Si", 22: "Ti", 38: "Sr"]
        var out: [Int: Double] = [:]
        for z in Set(numbers) { out[z] = bySymbol[symbols[z]!]! }
        return out
    }
}

@Suite("Thermal vibration amplitude per site against abTEM")
struct ThermalSitesPinsTests {
    @Test(arguments: ["T1", "T2", "T3", "T4"])
    func potentialsWithAbTEMsDisplacementsEqualAbTEM(id: String) throws {
        let c = ThermalPins.case(id)
        let g = c["gpts"] as! [Int]
        let fp = try ThermalPins.frozenPhonons(id)
        let potentials = try fp.potentials(nx: g[0], ny: g[1], sliceThickness: c["slice_thickness_requested"] as! Double,
                                           parameterization: .lobato(LobatoParameterization(piConstant: .abTEMFloat32)))
        #expect(potentials.count == (c["seeds"] as! [Int]).count)
        for (k, p) in potentials.enumerated() {
            #expect(p.slices.count == c["slices"] as! Int, "\(id) config \(k): slices")
            for (j, slice) in p.slices.enumerated() {
                let bound = relMax2(slice.values, ThermalPins.doubles("\(id).V.config\(k).slice\(j)"))
                #expect(bound <= 1e-11, "\(id) config \(k) slice \(j)")
            }
        }
    }

    /// σ per element resolves to abTEM's σ per atom exactly; the per-site cases hold different σ on one element.
    @Test func sigmaPerAtomEqualsAbTEMsResolution() throws {
        for id in ["T1", "T3"] {
            let s = try ThermalPins.structure(id)
            #expect(try FrozenPhonons.siteSigmas(of: s, sigmas: ThermalPins.elementSigmas(id))
                        == ThermalPins.doubles("\(id).site_sigmas"), "\(id)")
        }
        for id in ["T2", "T4"] {
            let s = try ThermalPins.structure(id)
            let site = ThermalPins.doubles("\(id).site_sigmas")
            #expect(site.count == s.atoms.count)
            let first = s.atoms[0].atomicNumber
            var ofFirst = Set<Double>()
            for (atom, sigma) in zip(s.atoms, site) where atom.atomicNumber == first { ofFirst.insert(sigma) }
            #expect(ofFirst.count > 1, "\(id): one element carries several σ")
            // The kit's own generator takes them.
            let fp = try FrozenPhonons(structure: s, siteSigmas: site, configurations: 2, seed: 1)
            #expect(fp.count == 2)
        }
    }

    @Test func sigmaAndBEqualAbTEM() {
        let p = ThermalPins.json["conversions"] as! [String: [Double]]
        for (sigma, b) in zip(p["sigma"]!, p["sigma_to_B"]!) {
            let kit = FrozenPhonons.bFactor(sigma: sigma)
            #expect(abs(kit - b) <= 1e-7 * abs(b), "sigma_to_B(\(sigma))")
        }
        for (b, sigma) in zip(p["B"]!, p["B_to_sigma"]!) {
            let kit = FrozenPhonons.sigma(bFactor: b)
            #expect(abs(kit - sigma) <= 1e-7 * abs(sigma), "B_to_sigma(\(b))")
        }
    }
}

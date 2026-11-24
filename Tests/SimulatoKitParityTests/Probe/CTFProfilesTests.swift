//
//  CTFProfilesTests.swift
//  Role: the CTF's radial profiles and its own closed forms against abTEM 1.0.10 (`CTF.profiles`,
//        `.scherzer_defocus`, `.point_resolution`, `.crossover_angle`), and what abTEM refuses (#102).
//

import Foundation
import Testing
import SimulatoCore
import SimulatoProbe

enum ProfilePins {
    struct Loaded: @unchecked Sendable {  // immutable after load
        let json: [String: Any]
    }

    static let loaded: Loaded = {
        let url = Pins3.directory.appendingPathComponent("ctf-profiles-pins.json")
        return Loaded(json: try! JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any])
    }()

    static var json: [String: Any] { loaded.json }

    static func `case`(_ id: String) -> [String: Any] { (json["cases"] as! [String: Any])[id] as! [String: Any] }

    static func ctf(_ id: String) throws -> ContrastTransferFunction {
        let c = Self.case(id)
        var a = Aberrations()
        let paths: [String: WritableKeyPath<Aberrations, Double>] = [
            "C10": \.C10, "C12": \.C12, "phi12": \.phi12, "C23": \.C23, "phi23": \.phi23, "C30": \.C30, "C50": \.C50,
        ]
        for (k, v) in c["aberrations"] as! [String: Double] { a[keyPath: paths[k]!] = v }
        return try ContrastTransferFunction(aberrations: a, semiangleCutoff: (c["cutoff_mrad"] as? Double) ?? .infinity,
                                            focalSpread: c["focal_spread"] as! Double,
                                            angularSpread: c["angular_spread"] as! Double)
    }
}

@Suite("CTF profiles against abTEM")
struct CTFProfilesTests {
    @Test(arguments: ["P1", "P2", "P3", "P4"])
    func profilesEqualAbTEM(id: String) throws {
        let c = ProfilePins.case(id)
        let p = try ProfilePins.ctf(id).profiles(energy: c["energy"] as! Double, points: c["points"] as! Int,
                                                 maxAngle: c["max_angle"] as? Double, phi: c["phi"] as! Double)
        let components = c["components"] as! [String]
        let points = c["points"] as! Int, sampling = c["sampling"] as! Double
        #expect(p.alpha.count == points)
        #expect(abs(p.sampling - sampling) <= 1e-14 * sampling, "\(id) sampling")
        #expect(relMax3(p.phase, c["phase"] as! [Double]) <= 1e-12, "\(id) phase")
        #expect(relMax3(p.transfer, c["ctf"] as! [Double]) <= 1e-12, "\(id) ctf")
        #expect((p.spatialEnvelope != nil) == components.contains("spatial envelope"), "\(id) spatial present")
        #expect((p.temporalEnvelope != nil) == components.contains("temporal envelope"), "\(id) temporal present")
        #expect((p.aperture != nil) == components.contains("aperture"), "\(id) aperture present")
        if let s = p.spatialEnvelope {
            #expect(relMax3(s, c["spatial_envelope"] as! [Double]) <= 1e-12, "\(id) spatial")
        }
        if let t = p.temporalEnvelope {
            #expect(relMax3(t, c["temporal_envelope"] as! [Double]) <= 1e-12, "\(id) temporal")
        }
        if let a = p.aperture {
            let pinned = c["aperture"] as! [Double]
            #expect(a == pinned, "\(id) aperture")
        }
    }

    @Test func instanceClosedFormsEqualAbTEM() throws {
        for row in ProfilePins.json["closed_forms"] as! [[String: Double]] {
            var a = Aberrations()
            a.Cs = row["Cs"]!
            let ctf = try ContrastTransferFunction(aberrations: a)
            let e = row["energy"]!
            let s = try ctf.scherzerDefocus(energy: e)
            let r = try ctf.pointResolution(energy: e)
            let x = try ctf.crossoverAngle(energy: e)
            #expect(abs(s - row["scherzer"]!) <= 1e-14 * abs(row["scherzer"]!), "Scherzer \(row)")
            #expect(abs(r - row["point_resolution"]!) <= 1e-14 * row["point_resolution"]!, "point resolution \(row)")
            #expect(abs(x - row["crossover_angle"]!) <= 1e-14 * row["crossover_angle"]!, "crossover \(row)")
        }
    }

    @Test func defaultsFollowAbTEM() throws {
        var a = Aberrations()
        a.Cs = 1e7
        let open = try ContrastTransferFunction(aberrations: a).profiles(energy: 200e3)
        #expect(open.alpha.count == 1000)
        #expect(open.maxAngle == 50)
        #expect(open.alpha.last == 50e-3)
        #expect(open.transfer == open.phase)
        let stopped = try ContrastTransferFunction(aberrations: a, semiangleCutoff: 20).profiles(energy: 200e3)
        #expect(stopped.maxAngle == 32)
        #expect(stopped.aperture?.last == 0)
        #expect(stopped.aperture?.first == 1)
    }

    @Test func refusesWhatAbTEMRefuses() throws {
        let none = try ContrastTransferFunction()
        #expect(throws: SimulatoError.self) { try none.scherzerDefocus(energy: 200e3) }
        #expect(throws: SimulatoError.self) { try none.crossoverAngle(energy: 200e3) }
        #expect(throws: SimulatoError.self) { try none.profiles(energy: 200e3, points: 1) }
        #expect(throws: SimulatoError.self) { try none.profiles(energy: 200e3, maxAngle: 0) }
        #expect(throws: SimulatoError.self) { try none.profiles(energy: 200e3, maxAngle: .nan) }
        #expect(throws: SimulatoError.self) { try none.profiles(energy: 200e3, phi: .infinity) }
        #expect(throws: SimulatoError.self) { try none.profiles(energy: 0) }
    }
}

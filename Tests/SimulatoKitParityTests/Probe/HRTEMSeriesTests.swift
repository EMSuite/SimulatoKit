//
//  HRTEMSeriesTests.swift
//  Role: the HRTEM image series (#88) against abTEM: a SrTiO₃ [001] slab with exit planes every four slices, the images
//        over four defocus values at each plane, and one series tiled (2, 3) (`tools/pins/hrtem-series/pins.py`).
//

import CryptoKit
import Foundation
import Testing
import SimulatoCore
import SimulatoMultislice
import SimulatoProbe

enum SeriesPins {
    static let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Fixtures")

    struct Loaded: @unchecked Sendable {  // immutable after load
        let json: [String: Any]
        let blob: Data
    }

    static let loaded: Loaded = {
        let json = try! JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent("hrtem-series-pins.json"))) as! [String: Any]
        let arrays = json["arrays"] as! [String: Any]
        let blob = try! Data(contentsOf: directory.appendingPathComponent(arrays["file"] as! String))
        let digest = SHA256.hash(data: blob).map { String(format: "%02x", $0) }.joined()
        precondition(digest == arrays["sha256"] as! String, "hrtem-series-arrays.bin does not match its sha256 in the pins")
        return Loaded(json: json, blob: blob)
    }()

    static var json: [String: Any] { loaded.json }

    static func doubles(_ name: String) -> [Double] {
        let entry = ((json["arrays"] as! [String: Any])["index"] as! [String: Any])[name] as! [String: Any]
        let offset = entry["offset"] as! Int, count = entry["count"] as! Int
        return loaded.blob.withUnsafeBytes { raw in
            (0..<count).map { i in
                Double(bitPattern: UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: offset + 8 * i, as: UInt64.self)))
            }
        }
    }

    /// The exit waves of the pins' slab, by the port: the incident wave, after slices 4 and 8, and after the last (10).
    static func exits(pi: LobatoParameterization.PiConstant) throws -> [Waves] {
        let a0 = 3.905
        let cell = try AtomicStructure(box: SIMD3(a0, a0, a0), atoms: [
            .init(atomicNumber: 38, position: SIMD3(0, 0, 0)), .init(atomicNumber: 22, position: SIMD3(0.5, 0.5, 0.5) * a0),
            .init(atomicNumber: 8, position: SIMD3(0.5, 0.5, 0) * a0), .init(atomicNumber: 8, position: SIMD3(0.5, 0, 0.5) * a0),
            .init(atomicNumber: 8, position: SIMD3(0, 0.5, 0.5) * a0),
        ])
        let p = try ProjectedPotential(structure: cell.repeated(2, 2, 5), nx: 64, ny: 64, sliceThickness: 1.9525,
                                       parameterization: .lobato(LobatoParameterization(piConstant: pi)))
        return try Multislice.run(.planeWave(beam: ElectronBeam(energy: 300e3), grid: p.grid), through: p,
                                  exitPlanes: .every(4))
    }

    static func ctf() throws -> ContrastTransferFunction {
        var lens = Aberrations()
        lens.C30 = json["Cs"] as! Double
        return try ContrastTransferFunction(aberrations: lens, semiangleCutoff: json["semiangle_cutoff"] as! Double,
                                            focalSpread: json["focal_spread"] as! Double)
    }
}

@Suite("HRTEM image series against abTEM")
struct HRTEMSeriesTests {
    @Test(arguments: [LobatoParameterization.PiConstant.abTEMFloat32, .exact])
    func seriesEqualsAbTEM(pi: LobatoParameterization.PiConstant) throws {
        let exits = try SeriesPins.exits(pi: pi)
        let defocus = SeriesPins.json["defocus"] as! [Double]
        #expect(exits.count == SeriesPins.json["planes"] as! Int)
        let bound = pi == .abTEMFloat32 ? 1e-10 : 1e-6
        let series = try SeriesPins.ctf().images(of: exits, defocus: defocus)
        for p in exits.indices {
            for d in defocus.indices {
                #expect(relMax3(series[p][d].values, SeriesPins.doubles("plane\(p).defocus\(d)")) <= bound, "plane \(p) defocus \(d)")
            }
        }
        let tile = SeriesPins.json["tile"] as! [Int]
        let tiled = try SeriesPins.ctf().images(of: [exits[exits.count - 1]], defocus: defocus, tile: (tile[0], tile[1]))
        for d in defocus.indices {
            #expect(tiled[0][d].nx == 64 * tile[0] && tiled[0][d].ny == 64 * tile[1])
            #expect(relMax3(tiled[0][d].values, SeriesPins.doubles("tiled.defocus\(d)")) <= bound, "tiled defocus \(d)")
        }
    }
}

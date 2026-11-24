//
//  KirklandPinsTests.swift
//  Role: the Kirkland parameterisation against abTEM 1.0.10 (tools/pins/kirkland/pins.py): f(k²) and the scaled
//        parameters for Z = 1 ... 103 under both π choices, the projected potential of one O, Al, Si, Ti and Sr atom
//        (1e-11 of the largest value, the Lobato pin's bound), and the exit wave of the Si [110] slab at the kit's
//        multislice pin (1e-10 with abTEM's float32 π, 1e-6 with the exact π).
//

import SimulatoKitTestSupport
import CryptoKit
import Foundation
import Testing
import SimulatoCore
import SimulatoMultislice

enum KirklandPins {
    static let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Fixtures")

    struct Loaded: @unchecked Sendable {  // immutable after load
        let json: [String: Any]
        let blob: Data
    }

    static let loaded: Loaded = {
        let json = try! JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent("kirkland-pins.json"))) as! [String: Any]
        let arrays = json["arrays"] as! [String: Any]
        let blob = try! Data(contentsOf: directory.appendingPathComponent(arrays["file"] as! String))
        let digest = SHA256.hash(data: blob).map { String(format: "%02x", $0) }.joined()
        precondition(digest == arrays["sha256"] as! String, "kirkland-arrays.bin does not match its sha256 in the pins")
        return Loaded(json: json, blob: blob)
    }()

    static var json: [String: Any] { loaded.json }

    static func doubles(_ name: String) -> (values: [Double], shape: [Int]) {
        let index = (json["arrays"] as! [String: Any])["index"] as! [String: Any]
        let entry = index[name] as! [String: Any]
        let offset = entry["offset"] as! Int, count = entry["count"] as! Int
        let kind = entry["kind"] as! String
        let n = kind == "c16" ? 2 * count : count
        let d: [Double] = loaded.blob.withUnsafeBytes { raw in
            (0..<n).map { i in
                Double(bitPattern: UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: offset + 8 * i, as: UInt64.self)))
            }
        }
        return (d, entry["shape"] as! [Int])
    }

    static func complex(_ name: String) -> ComplexField {
        let (d, shape) = doubles(name)
        return ComplexField.unchecked(nx: shape[0], ny: shape[1], real: stride(from: 0, to: d.count, by: 2).map { d[$0] },
                            imaginary: stride(from: 1, to: d.count, by: 2).map { d[$0] })
    }

    static let symbols: [String] = {
        let f = json["f"] as! [String: [String: Any]]
        return f.keys.sorted { (f[$0]!["z"] as! Int) < (f[$1]!["z"] as! Int) }
    }()
}

@Suite("Kirkland parameterisation against abTEM")
struct KirklandPinsTests {
    @Test func float32PiSquaredEqualsThePin() {
        #expect(KirklandParameterization.float32PiSquared == KirklandPins.json["pi2_kirkland_float32"] as! Double)
        #expect(PhysicalConstants.kappa == KirklandPins.json["kappa"] as! Double)
    }

    @Test func everyElementEqualsAbTEMUnderBothPiChoices() throws {
        let k2 = KirklandPins.json["f_k2"] as! [Double]
        let all = KirklandPins.json["f"] as! [String: [String: Any]]
        #expect(KirklandPins.symbols.count == 103)
        let abtemPi = KirklandParameterization(piConstant: .abTEMFloat32), exactPi = KirklandParameterization()
        for symbol in KirklandPins.symbols {
            let pin = all[symbol]!
            let z = pin["z"] as! Int
            let abtem = try abtemPi.projectedScatteringFactor(atomicNumber: z, k2: k2)
            let exact = try exactPi.projectedScatteringFactor(atomicNumber: z, k2: k2)
            #expect(relMax(abtem, pin["abtem"] as! [Double]) <= 1e-14, "\(symbol) abTEM π")
            #expect(relMax(exact, pin["exact"] as! [Double]) <= 1e-14, "\(symbol) exact π")
            let d = relMax(exact, pin["abtem"] as! [Double])
            #expect(d <= 1e-7 && d > 0, "\(symbol): the exact π against abTEM's: \(d)")
            let scaled = pin["scaled"] as! [[Double]]
            let s = try KirklandParameterization.scaledParameters(atomicNumber: z)
            for (row, mine) in zip(scaled, [s.a, s.b, s.c, s.d]) {
                #expect(relMax(mine, row) <= 1e-14, "\(symbol) scaled parameters")
            }
        }
    }

    @Test(arguments: ["O", "Al", "Si", "Ti", "Sr"])
    func singleAtomPotentialEqualsAbTEM(symbol: String) throws {
        let atoms = KirklandPins.json["atoms"] as! [String: Any]
        let box = atoms["box"] as! [Double], gpts = atoms["gpts"] as! [Int], position = atoms["position"] as! [Double]
        let pin = (atoms["cases"] as! [String: [String: Any]])[symbol]!
        let structure = try AtomicStructure(box: SIMD3(box[0], box[1], box[2]), atoms: [
            .init(atomicNumber: pin["z"] as! Int, position: SIMD3(position[0], position[1], position[2])),
        ])
        let potential = try ProjectedPotential(structure: structure, nx: gpts[0], ny: gpts[1],
                                               sliceThickness: atoms["slice_thickness"] as! Double,
                                               parameterization: .kirkland(KirklandParameterization(piConstant: .abTEMFloat32)))
        #expect(potential.slices.count == 1)
        let v = potential.slices[0].values
        #expect(relMax(v, KirklandPins.doubles("atom.\(symbol).V").values) <= 1e-11, "\(symbol) V")
        #expect(relDiff(v.reduce(0, +), pin["V_sum"] as! Double) <= 1e-11, "\(symbol) ΣV")
    }

    @Test(arguments: [(LobatoParameterization.PiConstant.abTEMFloat32, 1e-10), (.exact, 1e-6)])
    func siliconSlabExitWaveEqualsAbTEM(pi: LobatoParameterization.PiConstant, bound: Double) throws {
        let slab = SlabPins.slab
        let box = slab["box"] as! [Double], gpts = slab["gpts"] as! [Int]
        let numbers = slab["numbers"] as! [Int], positions = slab["positions"] as! [[Double]]
        let structure = try AtomicStructure(box: SIMD3(box[0], box[1], box[2]), atoms: zip(numbers, positions).map {
            AtomicStructure.Atom(atomicNumber: $0.0, position: SIMD3($0.1[0], $0.1[1], $0.1[2]))
        })
        let count = slab["slices"] as! Int
        let thicknesses = [Double](repeating: slab["slice_height"] as! Double, count: count)
        let potential = try Potential(structure: structure, nx: gpts[0], ny: gpts[1], sliceThicknesses: thicknesses,
                                      parameterization: .kirkland(KirklandParameterization(piConstant: pi)))
        let beam = try ElectronBeam(energy: slab["energy"] as! Double)
        let waves = try Multislice.run(Waves.planeWave(beam: beam, grid: potential.grid), through: potential)
        let reference = KirklandPins.complex("si110.exit")
        #expect(relMax(waves[0].field, reference) <= bound, "Si [110] Kirkland exit wave: \(relMax(waves[0].field, reference))")
    }
}

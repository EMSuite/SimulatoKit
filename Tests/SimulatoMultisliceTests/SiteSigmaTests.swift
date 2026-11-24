//
//  SiteSigmaTests.swift
//  Role: thermal vibration amplitude per site (#95): the kit's generator with a σ per atom, by Kolmogorov–Smirnov tests
//        of every site's displacements against the normal law of its σ; σ per element is σ per site to the last bit;
//        σ ↔ B = 8π²σ²; the refusals. abTEM's displacements and conversions are pinned in the parity target
//        (ThermalSitesPinsTests).
//

import Foundation
import Testing
import SimulatoCore
import SimulatoMultislice

/// The Kolmogorov–Smirnov statistic sup |F_n(x) − Φ(x)| of a sample against the standard normal law.
private func ksStatistic(_ sample: [Double]) -> Double {
    let x = sample.sorted()
    let n = Double(x.count)
    var d = 0.0
    for (i, v) in x.enumerated() {
        let phi = 0.5 * erfc(-v / 2.0.squareRoot())
        let above = Double(i + 1) / n - phi
        let below = phi - Double(i) / n
        d = max(d, max(above, below))
    }
    return d
}

/// The critical value of the one-sample KS statistic at significance 0.001 (asymptotic, 1.9495 / √n).
private func ksCritical(_ n: Int) -> Double { 1.9495 / Double(n).squareRoot() }

@Suite("Thermal vibration amplitude per site")
struct SiteSigmaTests {
    /// Four sites of one element with four different σ (one of them 0) over 2000 configurations: every site's
    /// displacements over the configurations and the three axes, divided by its σ, pass KS against N(0, 1); divided by
    /// the σ of another site they fail; the still site stays at zero.
    @Test func everySitePassesKSAtItsOwnSigma() throws {
        let sigmas = [0.05, 0.08, 0.12, 0.0]
        var atoms: [AtomicStructure.Atom] = []
        for i in sigmas.indices {
            atoms.append(AtomicStructure.Atom(atomicNumber: 14, position: SIMD3<Double>(1 + Double(i), 2, 3)))
        }
        let s = try AtomicStructure(box: SIMD3(8, 8, 8), atoms: atoms)
        let fp = try FrozenPhonons(structure: s, siteSigmas: sigmas, configurations: 2000, seed: 2026)
        for (i, sigma) in sigmas.enumerated() {
            var raw: [Double] = []
            for d in fp.displacements {
                raw.append(d[i].x)
                raw.append(d[i].y)
                raw.append(d[i].z)
            }
            if sigma == 0 {
                #expect(raw.allSatisfy { $0 == 0 }, "site \(i): σ = 0 stays still")
                continue
            }
            let own = ksStatistic(raw.map { $0 / sigma })
            #expect(own <= ksCritical(raw.count), "site \(i): KS D = \(own) at its σ = \(sigma)")
            let other = sigmas[(i + 1) % 3]
            let wrong = ksStatistic(raw.map { $0 / other })
            #expect(wrong > ksCritical(raw.count), "site \(i): KS rejects the σ of another site (D = \(wrong))")
        }
    }

    /// Many atoms in one configuration with σ alternating over three values and two elements: each σ class passes KS,
    /// so the σ follows the site, not the element.
    @Test func sigmaFollowsTheSiteNotTheElement() throws {
        let n = 12_000
        let ladder = [0.04, 0.07, 0.11]
        var atoms: [AtomicStructure.Atom] = []
        for i in 0..<n {
            let z: Int = i % 2 == 0 ? 8 : 22
            let x = Double(i % 100) * 0.1
            let y = Double(i / 100) * 0.08
            atoms.append(AtomicStructure.Atom(atomicNumber: z, position: SIMD3<Double>(x, y, 1)))
        }
        let s = try AtomicStructure(box: SIMD3(10, 10, 10), atoms: atoms)
        let siteSigmas = (0..<n).map { ladder[$0 % 3] }
        let fp = try FrozenPhonons(structure: s, siteSigmas: siteSigmas, configurations: 1, seed: 95, directions: [.x, .y])
        let d = fp.displacements[0]
        for (k, sigma) in ladder.enumerated() {
            var z: [Double] = []
            for i in stride(from: k, to: n, by: 3) {
                z.append(d[i].x / sigma)
                z.append(d[i].y / sigma)
            }
            let ks = ksStatistic(z)
            #expect(ks <= ksCritical(z.count), "σ = \(sigma): KS D = \(ks)")
        }
        #expect(d.allSatisfy { $0.z == 0 }, "z is not displaced")
    }

    /// σ per element is σ per site at each element's σ, to the last bit, for every choice of directions.
    @Test func perElementIsPerSiteToTheLastBit() throws {
        let species: [Int] = [38, 22, 8, 8, 8]
        var atoms: [AtomicStructure.Atom] = []
        for i in 0..<60 {
            let x = Double(i % 6)
            let y = Double(i / 6) * 0.7
            atoms.append(AtomicStructure.Atom(atomicNumber: species[i % 5], position: SIMD3<Double>(x, y, 1.5)))
        }
        let s = try AtomicStructure(box: SIMD3(8, 8, 4), atoms: atoms)
        let bySpecies: [Int: Double] = [38: 0.08, 22: 0.06, 8: 0.07]
        let resolved = try FrozenPhonons.siteSigmas(of: s, sigmas: bySpecies)
        #expect(resolved == s.atoms.map { bySpecies[$0.atomicNumber]! })
        for directions: FrozenPhonons.Directions in [.all, [.x, .y], .z] {
            let a = try FrozenPhonons(structure: s, sigmas: bySpecies, configurations: 3, seed: 7, directions: directions)
            let b = try FrozenPhonons(structure: s, siteSigmas: resolved, configurations: 3, seed: 7, directions: directions)
            #expect(a == b, "directions \(directions.rawValue)")
        }
    }

    /// B = 8π²σ² and back; σ per axis (abTEM's convention). abTEM's values are pinned in the parity target.
    @Test func sigmaAndBConvert() {
        #expect(FrozenPhonons.bFactor(sigma: 0) == 0)
        #expect(FrozenPhonons.sigma(bFactor: 0) == 0)
        let b = FrozenPhonons.bFactor(sigma: 0.075)
        let piSquared: Double = Double.pi * Double.pi
        let expected: Double = 8 * piSquared * (0.075 * 0.075)
        #expect(abs(b - expected) <= 1e-15 * b)
        for sigma in [1e-4, 0.05, 0.0707, 0.1, 0.3, 1.0] {
            let back = FrozenPhonons.sigma(bFactor: FrozenPhonons.bFactor(sigma: sigma))
            #expect(abs(back - sigma) <= 1e-15 * sigma, "σ = \(sigma)")
        }
        #expect(FrozenPhonons.sigma(bFactor: -1).isNaN, "a negative B gives NaN, as abTEM's")
    }

    @Test func refusals() throws {
        let s = try AtomicStructure(box: SIMD3(5, 5, 5), atoms: [
            .init(atomicNumber: 6, position: SIMD3(1, 1, 1)), .init(atomicNumber: 8, position: SIMD3(2, 2, 2)),
        ])
        #expect(throws: (any Error).self) { try FrozenPhonons(structure: s, siteSigmas: [0.1], configurations: 1, seed: 0) }
        #expect(throws: (any Error).self) {
            try FrozenPhonons(structure: s, siteSigmas: [0.1, 0.1, 0.1], configurations: 1, seed: 0)
        }
        #expect(throws: (any Error).self) { try FrozenPhonons(structure: s, siteSigmas: [0.1, -0.1], configurations: 1, seed: 0) }
        #expect(throws: (any Error).self) { try FrozenPhonons(structure: s, siteSigmas: [0.1, .nan], configurations: 1, seed: 0) }
        #expect(throws: (any Error).self) {
            try FrozenPhonons(structure: s, siteSigmas: [0.1, .infinity], configurations: 1, seed: 0)
        }
        #expect(throws: (any Error).self) { try FrozenPhonons(structure: s, siteSigmas: [0.1, 0.1], configurations: 0, seed: 0) }
        #expect(throws: (any Error).self) { try FrozenPhonons.siteSigmas(of: s, sigmas: [6: 0.1]) }
        #expect(throws: (any Error).self) {
            try FrozenPhonons(structure: s, sigmas: [6: 0.1, 8: .nan], configurations: 1, seed: 0)
        }
        let still = try FrozenPhonons(structure: s, siteSigmas: [0, 0], configurations: 2, seed: 1)
        for d in still.displacements {
            #expect(d == [SIMD3<Double>.zero, SIMD3<Double>.zero])
        }
    }
}

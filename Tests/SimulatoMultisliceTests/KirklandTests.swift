//
//  KirklandTests.swift
//  Role: the Kirkland parameterisation without fixtures: the scaled parameters from the table, f(0) in closed form, two
//        values abTEM 1.0.10 gives (tools/pins/kirkland/pins.py), the π choice, the k = 0 identity of a potential built
//        with it, and the refusals.
//

import SimulatoKitTestSupport
import Foundation
import Testing
import SimulatoCore
import SimulatoMultislice

@Suite("Kirkland parameterisation")
struct KirklandTests {
    @Test func scaledParametersFollowFromTheTable() throws {
        for z in [1, 8, 14, 38, 79, 103] {
            let p = try KirklandParameterization.parameters(atomicNumber: z)
            let s = try KirklandParameterization.scaledParameters(atomicNumber: z)
            #expect(p.a.count == 3 && p.b.count == 3 && p.c.count == 3 && p.d.count == 3)
            for i in 0..<3 {
                let kappa = PhysicalConstants.kappa
                #expect(relDiff(s.a[i], Double.pi * p.a[i] / kappa) <= 1e-15, "Z \(z) a\(i)")
                #expect(relDiff(s.b[i], 2 * Double.pi * p.b[i].squareRoot()) <= 1e-15, "Z \(z) b\(i)")
                #expect(relDiff(s.c[i], pow(Double.pi, 1.5) * p.c[i] / pow(p.d[i], 1.5) / kappa) <= 1e-15, "Z \(z) c\(i)")
                #expect(relDiff(s.d[i], Double.pi * Double.pi / p.d[i]) <= 1e-15, "Z \(z) d\(i)")
            }
        }
    }

    /// f(0) = Σ 4π a'/b'² + √(π/d') c' π/d', and f falls with k².
    @Test func scatteringFactorAtZeroIsTheClosedForm() throws {
        for z in [1, 6, 13, 14, 22, 38, 79, 92] {
            let s = try KirklandParameterization.scaledParameters(atomicNumber: z)
            var f0 = 0.0
            for i in 0..<3 {
                f0 += 4 * Double.pi * s.a[i] / (s.b[i] * s.b[i]) + (Double.pi / s.d[i]).squareRoot() * s.c[i] * Double.pi / s.d[i]
            }
            let p = KirklandParameterization()
            #expect(relDiff(try p.projectedScatteringFactor(atomicNumber: z, k2: 0), f0) <= 1e-14, "Z \(z)")
            let f = try p.projectedScatteringFactor(atomicNumber: z, k2: [0, 0.1, 1, 10])
            #expect(zip(f, f.dropFirst()).allSatisfy { $0 > $1 }, "Z \(z) falls with k²")
        }
    }

    /// abTEM 1.0.10's `KirklandParametrization().projected_scattering_factor("Si")` (float64), and the exact-π values.
    @Test func siliconEqualsAbTEM() throws {
        let abtem = KirklandParameterization(piConstant: .abTEMFloat32)
        let exact = KirklandParameterization()
        #expect(relDiff(try abtem.projectedScatteringFactor(atomicNumber: 14, k2: 0), 278.3742323275066) <= 1e-14)
        #expect(relDiff(try abtem.projectedScatteringFactor(atomicNumber: 14, k2: 1), 35.53873609423751) <= 1e-14)
        #expect(relDiff(try exact.projectedScatteringFactor(atomicNumber: 14, k2: 1), 35.53873775728653) <= 1e-14)
        #expect(KirklandParameterization.float32PiSquared == 9.86960506439209)
        // Kirkland and Lobato are two fits of the same atom: close, not equal (Si f(0): 278.37 against 279.41).
        let lobato = try LobatoParameterization().projectedScatteringFactor(atomicNumber: 14, k2: 0)
        #expect(relDiff(try exact.projectedScatteringFactor(atomicNumber: 14, k2: 0), lobato) < 0.01)
    }

    /// Truth: Σ V ΔxΔy = Σ f(0) on every slice, with Kirkland's f.
    @Test func potentialIntegratesToTheSumOfFAtZero() throws {
        let structure = try AtomicStructure(box: SIMD3(6, 6.4, 4), atoms: [
            .init(atomicNumber: 8, position: SIMD3(1.1, 2.0, 0.5)),
            .init(atomicNumber: 38, position: SIMD3(4.3, 3.9, 1.5)),
            .init(atomicNumber: 22, position: SIMD3(2.7, 5.1, 3.0)),
        ])
        let parameterization = Parameterization.kirkland(KirklandParameterization())
        let potential = try Potential(structure: structure, nx: 48, ny: 50, sliceThickness: 2,
                                      parameterization: parameterization)
        #expect(potential.parameterization == parameterization)
        let built = try potential.build()
        for (j, slice) in built.slices.enumerated() {
            let f0 = try potential.atoms(inSlice: j).map {
                try parameterization.projectedScatteringFactor(atomicNumber: $0.atomicNumber, k2: 0)
            }.reduce(0, +)
            let integral = slice.values.reduce(0, +) * built.grid.samplingX * built.grid.samplingY
            #expect(relDiff(integral, f0) <= 1e-12, "slice \(j)")
        }
        let lobato = try ProjectedPotential(structure: structure, nx: 48, ny: 50, sliceThickness: 2)
        #expect(relMax(built.slices[0].values, lobato.slices[0].values) > 1e-4, "the parameterisation reaches the potential")
    }

    @Test func refusals() {
        #expect(KirklandParameterization.supports(atomicNumber: 1) && KirklandParameterization.supports(atomicNumber: 103))
        #expect(!KirklandParameterization.supports(atomicNumber: 0) && !KirklandParameterization.supports(atomicNumber: 104))
        #expect(throws: SimulatoError.unsupportedElement(104)) { try KirklandParameterization.parameters(atomicNumber: 104) }
        #expect(throws: SimulatoError.unsupportedElement(0)) {
            try Parameterization.kirkland(KirklandParameterization()).projectedScatteringFactor(atomicNumber: 0, k2: 0)
        }
        #expect(throws: SimulatoError.unsupportedElement(110)) {
            try Potential(structure: AtomicStructure(box: SIMD3(4, 4, 4), atoms: [.init(atomicNumber: 110, position: .zero)]),
                          nx: 32, ny: 32, sliceThickness: 2, parameterization: .kirkland(KirklandParameterization()))
        }
    }
}

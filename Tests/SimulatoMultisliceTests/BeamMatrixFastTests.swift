//
//  BeamMatrixFastTests.swift
//  Role: one case per crystal of the beam matrix at 200 kV against abTEM 1.0.10 (tools/pins/beam-matrix/pins.py), on
//        every pull request; the parity target runs every case.
//

import Testing
import SimulatoKitTestSupport

@Suite("Beam matrix, one case per crystal")
struct BeamMatrixFastTests {
    @Test(arguments: BeamMatrix.fast)
    func caseAgreesWithAbTEM(_ index: Int) throws {
        let failures = try BeamMatrix.check(index)
        #expect(failures.isEmpty, "\(BeamMatrix.name(index)): \(failures.joined(separator: "; "))")
    }

    @Test func everyCrystalHasAFastCase() {
        #expect(BeamMatrix.fast.count >= 10, "\(BeamMatrix.fast.count) fast cases")
    }
}

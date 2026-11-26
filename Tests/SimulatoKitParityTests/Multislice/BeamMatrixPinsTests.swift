//
//  BeamMatrixPinsTests.swift
//  Role: every case of the beam matrix (tools/pins/beam-matrix/pins.py) against abTEM 1.0.10: eleven crystals, their
//        zone axes, 80, 200 and 300 kV. The hkl sets exact, the intensities within 1e-10 of the incident beam, no beam
//        the lattice forbids. The fast unit tests run one case per crystal of the same runner.
//

import Testing
import SimulatoKitTestSupport

@Suite("Beam matrix against abTEM")
struct BeamMatrixPinsTests {
    @Test(arguments: BeamMatrix.all)
    func caseAgreesWithAbTEM(_ index: Int) throws {
        let failures = try BeamMatrix.check(index)
        #expect(failures.isEmpty, "\(BeamMatrix.name(index)): \(failures.joined(separator: "; "))")
    }
}

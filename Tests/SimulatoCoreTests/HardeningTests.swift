//
//  HardeningTests.swift
//  Role: inputs that used to trap late in SimulatoCore (huge discs, tilings and resamples) now throw their typed errors.
//

import Foundation
import Testing
import SimulatoCore

@Suite("Core hardening")
struct CoreHardeningTests {
    func image() -> RealField {
        RealField.unchecked(nx: 4, ny: 4, values: [Double](repeating: 1, count: 16), samplingX: 1, samplingY: 1)
    }

    @Test func hugeDiscIsRefused() throws {
        let f = image()
        #expect(throws: ImageOperationError.self) { try f.integrateDisc(x: 1, y: 1, radius: 1e300) }
        #expect(throws: ImageOperationError.self) { try f.integrateDisc(x: 1e300, y: 1, radius: 1) }
        #expect(throws: ImageOperationError.self) { try f.integrateDisc(x: 1, y: -1e300, radius: 1) }
        // a valid disc is unchanged: the 3 x 3 box of ones sums to at most 9
        #expect(try f.integrateDisc(x: 2, y: 2, radius: 1) > 0)
    }

    @Test func hugeTilingIsRefused() throws {
        let f = image()
        #expect(throws: ImageOperationError.invalidRepetitions(.max, 2)) { try f.tiled(repetitionsX: .max, repetitionsY: 2) }
        #expect(throws: ImageOperationError.self) { try f.tiled(repetitionsX: 1 << 20, repetitionsY: 1 << 20) }
        #expect(try f.tiled(repetitionsX: 2, repetitionsY: 3).values.count == 8 * 12)
    }

    @Test func hugeResampleIsRefused() throws {
        let grid = try Grid(nx: 4, ny: 4, extentX: 1, extentY: 1)
        let w = Waves.unchecked(field: ComplexField(nx: 4, ny: 4), grid: grid, beam: try ElectronBeam(energy: 200e3), normalization: .values)
        #expect(throws: SimulatoError.self) { try w.resampled(rows: 1 << 40, columns: 1 << 30) }
        #expect(throws: SimulatoError.self) { try w.resampled(rows: 1 << 20, columns: 1 << 20) }
        #expect(try w.resampled(rows: 8, columns: 8).grid.nx == 8)
    }

    /// #59: nx · ny (and its 16-byte values) overflowed `Grid.count` and every array size after it.
    @Test func gridWhoseValuesOverflowIsRefused() throws {
        #expect(throws: SimulatoError.gridTooLarge(rows: 1 << 32, columns: 1 << 32, required: .max, limit: .max)) {
            try Grid(nx: 1 << 32, ny: 1 << 32, extentX: 1, extentY: 1)
        }
        #expect(throws: SimulatoError.self) { try Grid(nx: 1 << 31, ny: 1 << 30, extentX: 1, extentY: 1) }
        #expect(try Grid(nx: 1 << 20, ny: 1 << 20, extentX: 1, extentY: 1).count == 1 << 40)
    }

    /// #80: the public initialisers threw nothing and trapped on values that do not fill the shape.
    @Test func fieldsAndWavesRefuseValuesThatDoNotFillTheShape() throws {
        #expect(throws: SimulatoError.self) { try ComplexField(nx: 2, ny: 3, real: [Double](repeating: 0, count: 5),
                                                                imaginary: [Double](repeating: 0, count: 6)) }
        #expect(throws: SimulatoError.self) { try ComplexField(nx: 2, ny: 3, real: [Double](repeating: 0, count: 6),
                                                                imaginary: []) }
        #expect(throws: SimulatoError.self) { try ComplexField(nx: -2, ny: -3, real: [Double](repeating: 0, count: 6),
                                                                imaginary: [Double](repeating: 0, count: 6)) }
        #expect(throws: SimulatoError.self) { try ComplexField(nx: 1 << 40, ny: 1 << 40, real: [], imaginary: []) }
        #expect(throws: SimulatoError.self) { try RealField(nx: 2, ny: 2, values: [1, 2, 3], samplingX: 1, samplingY: 1) }
        let grid = try Grid(nx: 4, ny: 4, extentX: 1, extentY: 1)
        let beam = try ElectronBeam(energy: 200e3)
        #expect(throws: SimulatoError.self) {
            try Waves(field: ComplexField(nx: 4, ny: 2), grid: grid, beam: beam, normalization: .values)
        }
        #expect(try Waves(field: ComplexField(nx: 4, ny: 4), grid: grid, beam: beam, normalization: .values).grid == grid)
        #expect(try ComplexField(nx: 1, ny: 2, real: [1, 2], imaginary: [0, 0])[0, 1].re == 2)
    }

    /// #80: `repeated` trapped on a count below 1 and on an atom count or a box beyond what an Int or a Double holds.
    @Test func repeatRefusesCountsItCannotHold() throws {
        let s = try AtomicStructure(box: SIMD3(1, 1, 1), atoms: [.init(atomicNumber: 1, position: .zero)])
        #expect(throws: SimulatoError.self) { try s.repeated(0, 1, 1) }
        #expect(throws: SimulatoError.self) { try s.repeated(1, -1, 1) }
        #expect(throws: SimulatoError.self) { try s.repeated(1 << 31, 1 << 31, 1 << 31) }
        let huge = try AtomicStructure(box: SIMD3(1e308, 1, 1), atoms: [])
        #expect(throws: SimulatoError.self) { try huge.repeated(10, 1, 1) }
        #expect(try s.repeated(2, 1, 1).atoms.count == 2)
    }
}

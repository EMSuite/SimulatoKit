//
//  ScanAzimuthalAverageTests.swift
//  Role: the azimuthal average over a pixelated scan (#9; abTEM's `DiffractionPatterns.azimuthal_average` on a scan's
//        patterns, the scan axes as ensemble axes): every position equals the single-pattern average (pinned against
//        abTEM in AzimuthalAverageTests) bit for bit, on a line scan and a grid scan; a segmented line scan gives line
//        profiles through `integrate`; and the refusals.
//

import Testing
import SimulatoCore
import SimulatoDetector
import SimulatoKitTestSupport
import SimulatoMultislice
import SimulatoProbe

@Suite("Azimuthal average over a scan")
struct ScanAzimuthalAverageTests {
    /// Al 1 x 1 x 2 on 32², 200 keV, 20 mrad.
    static func setup() throws -> (probe: Probe, potential: Potential) {
        let potential = try Potential(structure: aluminiumStructure(cells: (1, 1, 2)), nx: 32, ny: 32, sliceThickness: 2.025)
        let probe = try Probe(beam: ElectronBeam(energy: 200e3), grid: potential.grid, aperture: Aperture(semiangleCutoff: 20))
        return (probe, potential)
    }

    static func single(_ pattern: [Double], _ shape: [Int], _ probe: Probe,
                       weighting: AzimuthalWeighting = .step) throws -> ReciprocalSpaceLineProfiles {
        try RealField(nx: shape[0], ny: shape[1], values: pattern, samplingX: probe.grid.reciprocalSamplingX,
                      samplingY: probe.grid.reciprocalSamplingY)
            .azimuthalAverage(beam: probe.beam, weighting: weighting)
    }

    @Test func aPixelatedLineScanAveragesEveryPosition() throws {
        let (probe, potential) = try Self.setup()
        let scan = try LineScan(start: SIMD2(0, 0), end: SIMD2(4.05, 2), gpts: 4)
        let m = try STEM.scan(probe: probe, potentials: [potential], scan: scan, detectors: [PixelatedDetector()])[0]
        let profiles = try m.azimuthalAverage(grid: probe.grid, beam: probe.beam)
        #expect(profiles.ensembleShape == [4])
        let first = try Self.single(m[0], m.valueShape, probe)
        #expect(profiles.count == first.count && profiles.count > 0 && profiles.sampling == first.sampling)
        for q in 0..<4 {
            let expected = try Self.single(m[q], m.valueShape, probe)
            #expect(Array(profiles.values[(q * profiles.count)..<((q + 1) * profiles.count)]) == expected.values)
        }
    }

    @Test func aPixelatedGridScanKeepsItsTwoAxes() throws {
        let (probe, potential) = try Self.setup()
        let scan = try GridScan(start: SIMD2(0, 0), end: SIMD2(4.05, 4.05), nx: 3, ny: 2)
        let m = try STEM.scan(probe: probe, potentials: [potential], scan: scan, detectors: [PixelatedDetector()])[0]
        let profiles = try m.azimuthalAverage(grid: probe.grid, beam: probe.beam, weighting: .gaussian)
        #expect(profiles.ensembleShape == [3, 2])
        for ix in 0..<3 {
            for iy in 0..<2 {
                let q = ix * 2 + iy
                let expected = try Self.single(m[ix, iy], m.valueShape, probe, weighting: .gaussian)
                #expect(Array(profiles.values[(q * profiles.count)..<((q + 1) * profiles.count)]) == expected.values)
            }
        }
    }

    @Test func aSegmentedLineScanGivesLineProfilesThroughIntegrate() throws {
        let (probe, potential) = try Self.setup()
        let scan = try LineScan(start: SIMD2(0, 0), end: SIMD2(4.05, 2), gpts: 5)
        let detector = try SegmentedDetector(radialBins: 2, azimuthalBins: 4, inner: 0, outer: 20)
        let m = try STEM.scan(probe: probe, potentials: [potential], scan: scan, detectors: [detector])[0]
        #expect(m.scanShape == [5] && m.valueShape == [2, 4])
        let integrated = try m.integrate(detector: detector)
        let profile = try integrated.lineProfiles(for: scan)
        #expect(profile.count == 5 && profile.values == integrated.values)
        let (dx, dy) = try m.differentials(direction1: ([0], [2]), direction2: ([1], [3]))
        #expect(try dx.lineProfiles(for: scan).count == 5 && dy.scanShape == [5])
    }

    @Test func onlyAPixelatedScanIsAveraged() throws {
        let (probe, potential) = try Self.setup()
        let scan = try LineScan(start: SIMD2(0, 0), end: SIMD2(4.05, 2), gpts: 3)
        let annular = try STEM.scan(probe: probe, potentials: [potential], scan: scan,
                                    detectors: [AnnularDetector(inner: 0, outer: 20)])[0]
        #expect(throws: SimulatoError.self) { try annular.azimuthalAverage(grid: probe.grid, beam: probe.beam) }
        let tooLarge = try ScanMeasurement(scanShape: [1], valueShape: [64, 64], values: [Double](repeating: 0, count: 64 * 64))
        #expect(throws: SimulatoError.self) { try tooLarge.azimuthalAverage(grid: probe.grid, beam: probe.beam) }
        let pixelated = try STEM.scan(probe: probe, potentials: [potential], scan: scan, detectors: [PixelatedDetector()])[0]
        #expect(throws: SimulatoError.self) {
            try pixelated.azimuthalAverage(grid: probe.grid, beam: probe.beam, radialSampling: 0)
        }
    }
}

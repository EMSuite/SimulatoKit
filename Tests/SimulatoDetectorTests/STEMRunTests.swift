//
//  STEMRunTests.swift
//  Role: a STEM scan that stops and continues: a cancelled run keeps its finished positions and configurations, a
//        continued run equals an uninterrupted one bit for bit on the CPU, progress counts positions and configurations
//        and reaches 1, and a handler changes no number.
//

import Testing
import SimulatoCore
import SimulatoDetector
import SimulatoKitTestSupport
import SimulatoMultislice
import SimulatoProbe

@Suite("STEM run: stop and continue")
struct STEMRunTests {
    /// Three frozen-phonon configurations of Al 2 x 2 x 2 on 32², a 3 x 3 scan, a bright-field and an annular detector.
    static func setup() throws -> (Probe, [Potential], GridScan, [any Detector]) {
        let structure = try aluminiumStructure(cells: (2, 2, 2))
        let phonons = try FrozenPhonons(structure: structure, sigmas: [13: 0.1], configurations: 3, seed: 7)
        let potentials = try phonons.lazyPotentials(nx: 32, ny: 32, sliceThickness: 2.025)
        let probe = try Probe(beam: ElectronBeam(energy: 200e3), grid: potentials[0].grid,
                              aperture: Aperture(semiangleCutoff: 20))
        let scan = try GridScan(start: SIMD2(0, 0), end: SIMD2(4.05, 4.05), nx: 3, ny: 3)
        let detectors: [any Detector] = [try AnnularDetector(inner: 0, outer: 20), try AnnularDetector(inner: 25, outer: 45)]
        return (probe, potentials, scan, detectors)
    }

    static let twoThreads = ExecutionOptions(maximumCPUThreads: 2, progressInterval: .zero)

    /// Runs `run` in a task that cancels itself at the first progress call `stop` accepts; returns the run as it stopped.
    static func stopped(_ run: STEMRun, at stop: @escaping @Sendable (SimulationProgress) -> Bool) async throws -> STEMRun {
        let task = Task {
            var r = run
            do {
                _ = try r.run { p in if stop(p) { withUnsafeCurrentTask { $0?.cancel() } } }
                Issue.record("the run was not cancelled")
            } catch is CancellationError {}
            return r
        }
        return try await task.value
    }

    @Test func aContinuedRunEqualsAnUninterruptedOne() async throws {
        let (probe, potentials, scan, detectors) = try Self.setup()
        let whole = try STEM.scan(probe: probe, potentials: potentials, scan: scan, detectors: detectors)
        let fresh = try STEMRun(probe: probe, potentials: potentials, scan: scan, detectors: detectors,
                                options: Self.twoThreads)
        // Stopped in configuration 1 after two rounds of two positions.
        var run = try await Self.stopped(fresh) { $0.stage == .scanning && $0.configuration == 1 && $0.completed == 4 }
        #expect(run.completedConfigurations == 1 && run.completedPositions == 4 && !run.isComplete)
        // The partial measurement: two configurations' mean at the first four positions, one configuration's elsewhere.
        let partial = run.measurements()
        #expect(partial.count == 2 && partial[0].values.allSatisfy { $0 > 0 })
        // Stopped again in configuration 2, then continued to the end.
        run = try await Self.stopped(run) { $0.stage == .scanning && $0.configuration == 2 && $0.completed == 2 }
        #expect(run.completedConfigurations == 2 && run.completedPositions == 2)
        let log = ProgressLog()
        let continued = try run.run(progress: log.handler)
        #expect(run.isComplete && run.completedPositions == 0)
        for (a, b) in zip(continued, whole) {
            #expect(a.values == b.values && a.scanShape == b.scanShape && a.valueShape == b.valueShape)
        }
        #expect(run.measurements().map(\.values) == whole.map(\.values))
        // The continued run's progress picks up where it stopped and reaches 1.
        let fractions = log.all.compactMap(\.fraction)
        #expect(fractions.first! >= (2 * 9 + 2) / 27.0 && fractions.last == 1)
        #expect(zip(fractions, fractions.dropFirst()).allSatisfy { $0 <= $1 })
        // A finished run has nothing left to do.
        let again = try run.run()
        #expect(again.map(\.values) == whole.map(\.values))
    }

    @Test func aStoppedStaticScanKeepsItsFinishedPositions() async throws {
        let (probe, potentials, scan, detectors) = try Self.setup()
        let single = [potentials[0]]
        let whole = try STEM.scan(probe: probe, potentials: single, scan: scan, detectors: detectors)
        let fresh = try STEMRun(probe: probe, potentials: single, scan: scan, detectors: detectors, options: Self.twoThreads)
        let run = try await Self.stopped(fresh) { $0.stage == .scanning && $0.completed == 6 }
        #expect(run.completedConfigurations == 0 && run.completedPositions == 6)
        for (a, b) in zip(run.measurements(), whole) {
            #expect(Array(a.values[0..<6]) == Array(b.values[0..<6]), "finished positions are final")
            #expect(a.values[6...].allSatisfy { $0 == 0 }, "unfinished positions are empty")
        }
    }

    @Test func progressCountsPositionsAndConfigurations() throws {
        let (probe, potentials, scan, detectors) = try Self.setup()
        let plain = try STEM.scan(probe: probe, potentials: potentials.map { try $0.build() }, scan: scan, detectors: detectors)
        let log = ProgressLog()
        let reported = try STEM.scan(probe: probe, potentials: potentials.map { try $0.build() }, scan: scan,
                                     detectors: detectors, options: Self.twoThreads, progress: log.handler)
        #expect(plain.map(\.values) == reported.map(\.values))
        let calls = log.all
        // Per configuration: preparing (no fraction), then the rounds 2, 4, 6, 8, 9.
        #expect(calls.count == 3 * 6)
        for k in 0..<3 {
            let part = Array(calls[(6 * k)..<(6 * k + 6)])
            #expect(part[0].stage == .preparing && part[0].fraction == nil && part[0].configuration == k)
            #expect(part.dropFirst().map(\.completed) == [2, 4, 6, 8, 9])
            #expect(part.allSatisfy { $0.configurationCount == 3 && $0.total == 9 && $0.unit == .positions })
        }
        #expect(calls.last?.fraction == 1)
        // One potential: no configuration in the progress.
        let one = ProgressLog()
        _ = try STEM.scan(probe: probe, potentials: [potentials[0]], scan: scan, detectors: detectors,
                          options: Self.twoThreads, progress: one.handler)
        #expect(one.all.allSatisfy { $0.configuration == nil && $0.configurationCount == nil })
    }
}

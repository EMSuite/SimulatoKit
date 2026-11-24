//
//  ProgressAndCancellationTests.swift
//  Role: a multislice run reports its progress and stops when its task is cancelled: a handler changes no number,
//        no call arrives after a cancel, `CancellationError` comes within one slice (CPU) or chunk (Metal), the fraction
//        is absent while the run prepares, and two Metal runs interleave between chunks.
//

import Dispatch
import Synchronization
import Testing
import SimulatoCore
import SimulatoMultislice
import SimulatoProbe
import SimulatoKitTestSupport

@Suite("Progress and cancellation")
struct ProgressAndCancellationTests {
    /// Al, 2 x 2 x 10 cells on 64², 20 slices, and a plane wave.
    static func crystal() throws -> (Potential, Waves) {
        let lazy = try Potential(structure: aluminiumStructure(cells: (2, 2, 10)), nx: 64, ny: 64, sliceThickness: 2.025)
        let beam = try ElectronBeam(energy: 200e3)
        return (lazy, Waves.planeWave(beam: beam, grid: lazy.grid))
    }

    static let everyCall = ExecutionOptions(progressInterval: .zero)

    @Test func aHandlerChangesNoNumberAndCountsEverySlice() throws {
        let (lazy, wave) = try Self.crystal()
        let built = try lazy.build()
        let plain = try Multislice.run(wave, through: built, exitPlanes: .every(5))
        let log = ProgressLog()
        let reported = try Multislice.run(wave, through: built, exitPlanes: .every(5), options: Self.everyCall,
                                          progress: log.handler)
        #expect(plain.map(\.field) == reported.map(\.field))
        let calls = log.all
        #expect(calls.first?.stage == .preparing && calls.first?.fraction == nil)
        #expect(calls.dropFirst().map(\.completed) == Array(1...20))
        #expect(calls.dropFirst().allSatisfy { $0.stage == .propagating && $0.unit == .slices && $0.total == 20 })
        #expect(calls.last?.fraction == 1)
        // The same through the unbuilt potential, and for a batch of waves.
        #expect(try Multislice.run(wave, through: lazy, progress: log.handler).map(\.field) == [plain.last!.field])
        let probe = try Probe(beam: wave.beam, grid: lazy.grid, aperture: Aperture(semiangleCutoff: 20))
        let incidents = try (0..<5).map { try probe.build(at: SIMD2(0.3 * Double($0), 0.2 * Double($0))) }
        let batchLog = ProgressLog()
        let batch = try Multislice.runBatch(incidents, through: built, options: ExecutionOptions(maximumCPUThreads: 2,
                                            progressInterval: .zero), progress: batchLog.handler)
        #expect(batch.map(\.field) == (try Multislice.runBatch(incidents, through: built)).map(\.field))
        #expect(batchLog.all.dropFirst().map(\.completed) == [2, 4, 5], "rounds of one wave per thread")
    }

    @Test func aCancelStopsTheCPURunAtTheNextSlice() async throws {
        let (lazy, wave) = try Self.crystal()
        let built = try lazy.build()
        let log = ProgressLog()
        let task = Task {
            try Multislice.run(wave, through: built, options: Self.everyCall) { p in
                log.calls.withLock { $0.append(p) }
                if p.stage == .propagating && p.completed == 3 { withUnsafeCurrentTask { $0?.cancel() } }
            }
        }
        let result = await task.result
        #expect(throws: CancellationError.self) { try result.get() }
        #expect(log.all.last?.completed == 3, "no call after the cancel")
        // A batch stops after the round under way.
        let probe = try Probe(beam: wave.beam, grid: lazy.grid, aperture: Aperture(semiangleCutoff: 20))
        let incidents = try (0..<6).map { try probe.build(at: SIMD2(0.3 * Double($0), 0.2 * Double($0))) }
        let batchLog = ProgressLog()
        let batchTask = Task {
            try Multislice.runBatch(incidents, through: built, options: ExecutionOptions(maximumCPUThreads: 2,
                                                                                         progressInterval: .zero)) { p in
                batchLog.calls.withLock { $0.append(p) }
                if p.completed == 2 { withUnsafeCurrentTask { $0?.cancel() } }
            }
        }
        let batchResult = await batchTask.result
        #expect(throws: CancellationError.self) { try batchResult.get() }
        #expect(batchLog.all.last?.completed == 2)
        // A task cancelled before the run starts does no work.
        let early = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try Multislice.run(wave, through: built)
        }
        let earlyResult = await early.result
        #expect(throws: CancellationError.self) { try earlyResult.get() }
    }

    @Test(.enabled(if: Multislice.metalIsAvailable, "no Metal device"))
    func metalWithAHandlerStaysInsideTheRegisteredTolerance() throws {
        let (lazy, wave) = try Self.crystal()
        let cpu = try Multislice.run(wave, through: lazy, exitPlanes: .every(7))
        let plain = try Multislice.run(wave, through: lazy, exitPlanes: .every(7), backend: .metal)
        let log = ProgressLog()
        let reported = try Multislice.run(wave, through: lazy, exitPlanes: .every(7), backend: .metal,
                                          options: Self.everyCall, progress: log.handler)
        // The same kernels; only the chunk boundaries (set by timing) may differ.
        for (a, b) in zip(reported, plain) { #expect(relMax(a.field, b.field) <= 1e-6) }
        for (a, b) in zip(reported, cpu) { #expect(relMax(a.field, b.field) <= MetalAccuracy.registered.wave) }
        let calls = log.all
        #expect(calls.first?.stage == .preparing && calls.first?.fraction == nil, "no fraction while the program compiles")
        #expect(calls.count >= 2 && calls.last?.completed == 20 && calls.last?.fraction == 1)
        #expect(zip(calls.dropFirst(), calls.dropFirst(2)).allSatisfy { $0.completed < $1.completed })
    }

    @Test(.enabled(if: Multislice.metalIsAvailable, "no Metal device"))
    func aCancelStopsTheMetalRunAtTheNextChunk() async throws {
        let (lazy, wave) = try Self.crystal()
        let log = ProgressLog()
        let task = Task {
            try Multislice.run(wave, through: lazy, backend: .metal, options: Self.everyCall) { p in
                log.calls.withLock { $0.append(p) }
                if p.stage == .propagating { withUnsafeCurrentTask { $0?.cancel() } }
            }
        }
        let result = await task.result
        #expect(throws: CancellationError.self) { try result.get() }
        let propagating = log.all.filter { $0.stage == .propagating }
        #expect(propagating.count == 1 && propagating[0].completed < 20, "one chunk, then the cancel")
    }

    /// While a long Metal run is between chunks (its handler runs with no lock held) a second Metal run finishes.
    @Test(.enabled(if: Multislice.metalIsAvailable, "no Metal device"))
    func twoMetalRunsInterleave() throws {
        let (lazy, wave) = try Self.crystal()
        let short = try Potential(structure: aluminiumStructure(cells: (2, 2, 1)), nx: 64, ny: 64, sliceThickness: 2.025)
        let started = ProgressLog(), finished = ProgressLog()
        let reference = try Multislice.run(wave, through: short, backend: .metal)
        let second = DispatchGroup()
        _ = try Multislice.run(wave, through: lazy, backend: .metal, options: Self.everyCall) { p in
            // One second run only: the guard is on starting it, so a slow runner past the wait starts no third.
            guard p.stage == .propagating, started.all.isEmpty else { return }
            started.calls.withLock { $0.append(p) }
            second.enter()
            DispatchQueue.global().async {
                if let out = try? Multislice.run(wave, through: short, backend: .metal),
                   zip(out, reference).allSatisfy({ relMax($0.field, $1.field) <= 1e-6 }) {
                    finished.calls.withLock { $0.append(p) }
                }
                second.leave()
            }
            // Deadlock-free means the second run completes while the first is parked here (74 s seen on a busy
            // runner); the bound only keeps a real deadlock from hanging CI.
            _ = second.wait(timeout: .now() + 600)
        }
        _ = second.wait(timeout: .now() + 600)
        #expect(started.all.count == 1)
        #expect(finished.all.count == 1, "the second run finished inside the first run's handler")
    }
}

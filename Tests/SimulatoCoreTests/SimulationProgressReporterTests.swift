//
//  ProgressReporterTests.swift
//  Role: the progress reporter every run loop calls: the first call always, then at most one per interval (ten a second
//        by default), and none once the task is cancelled.
//

import Synchronization
import Testing
import SimulatoCore

/// The calls a handler received, shared with the handler.
final class ProgressLog: Sendable {
    let calls = Mutex<[SimulationProgress]>([])
    var all: [SimulationProgress] { calls.withLock { $0 } }
    var handler: SimulationProgressHandler { { p in self.calls.withLock { $0.append(p) } } }
}

/// A clock the test moves by hand.
final class HandClock {
    let start = ContinuousClock.now
    var elapsed = Duration.zero
    var now: ContinuousClock.Instant { start.advanced(by: elapsed) }
}

@Suite("Progress reporter")
struct ProgressReporterTests {
    static func step(_ k: Int) -> SimulationProgress {
        SimulationProgress(stage: .propagating, completed: k, total: 100, unit: .slices, fraction: Double(k) / 100)
    }

    @Test func atMostOneCallPerInterval() {
        let log = ProgressLog()
        let clock = HandClock()
        var reporter = SimulationProgressReporter(log.handler, interval: .milliseconds(100), now: { clock.now })
        // Calls every 30 ms for 1 s: delivered at 0, 120, 240, ... (the first call after each 100 ms), 9 in all.
        for k in 0...33 {
            clock.elapsed = .milliseconds(30 * k)
            reporter.report(Self.step(k))
        }
        #expect(log.all.map(\.completed) == [0, 4, 8, 12, 16, 20, 24, 28, 32])
        // At most ten in any second, whatever the pace.
        let log2 = ProgressLog()
        var fast = SimulationProgressReporter(log2.handler, interval: .milliseconds(100), now: { clock.now })
        for k in 0..<10_000 {
            clock.elapsed = .microseconds(100 * k)
            fast.report(Self.step(k))
        }
        #expect(log2.all.count == 10)
        // Interval zero: every call.
        let log3 = ProgressLog()
        var every = SimulationProgressReporter(log3.handler, interval: .zero, now: { clock.now })
        for k in 0..<5 { every.report(Self.step(k)) }
        #expect(log3.all.count == 5)
        // No handler: nothing to call, and the progress is not even built.
        var none = SimulationProgressReporter(nil, interval: .zero)
        #expect(!none.isActive)
        none.report(Self.step(0))
    }

    @Test func noCallOnceTheTaskIsCancelled() async {
        let log = ProgressLog()
        let task = Task {
            var reporter = SimulationProgressReporter(log.handler, interval: .zero)
            reporter.report(Self.step(1))
            withUnsafeCurrentTask { $0?.cancel() }
            reporter.report(Self.step(2))
        }
        await task.value
        #expect(log.all.map(\.completed) == [1])
    }

    @Test func theFractionIsClampedAndMayBeAbsent() {
        #expect(SimulationProgress(stage: .preparing, completed: 0, total: 1, unit: .slices, fraction: nil).fraction == nil)
        #expect(SimulationProgress(stage: .scanning, completed: 0, total: 1, unit: .positions, fraction: 1.5).fraction == 1)
        #expect(SimulationProgress(stage: .scanning, completed: 0, total: 1, unit: .positions, fraction: -1).fraction == 0)
    }
}

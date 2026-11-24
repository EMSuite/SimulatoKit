//
//  WorkCounter.swift
//  Role: for the tests, a count of the expensive steps a run takes (potentials prepared, FFT plans made, probe spectra
//        computed), bound to a task with `WorkCounter.$current.withValue`, so a test can show a scan does a step once
//        and not once per chunk or position (#61). Counted on the calling thread only.
//

import Synchronization

/// Counts of the expensive steps of the runs made while it is bound (`WorkCounter.$current.withValue`).
package final class WorkCounter: Sendable {
    /// The counter bound to the current task, if any.
    @TaskLocal package static var current: WorkCounter?

    private let preparations = Mutex(0)
    private let fftPlans = Mutex(0)
    private let probeSpectra = Mutex(0)

    package init() {}

    /// Potentials made ready for propagation: transmission functions built (CPU) or a potential prepared on the GPU.
    package var preparationCount: Int { preparations.withLock { $0 } }
    /// FFT plans made for CPU batch workers.
    package var fftPlanCount: Int { fftPlans.withLock { $0 } }
    /// Probe spectra (aperture and aberration phase over the grid) computed for building probes.
    package var probeSpectrumCount: Int { probeSpectra.withLock { $0 } }

    package func countPreparation() { preparations.withLock { $0 += 1 } }
    package func countFFTPlan() { fftPlans.withLock { $0 += 1 } }
    package func countProbeSpectrum() { probeSpectra.withLock { $0 += 1 } }
}

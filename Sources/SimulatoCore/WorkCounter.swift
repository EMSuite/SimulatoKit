//
//  WorkCounter.swift
//  Role: for the tests, a count of the expensive steps a run takes (potentials prepared, FFT plans made, probe spectra
//        computed, slice potentials and transmissions made), bound to a task with `WorkCounter.$current.withValue`, so a
//        test can show a scan does a step once and not once per chunk, position or repeated slice (#61, #122). Counted
//        on the calling thread only.
//

import Synchronization

/// Counts of the expensive steps of the runs made while it is bound (`WorkCounter.$current.withValue`).
package final class WorkCounter: Sendable {
    /// The counter bound to the current task, if any.
    @TaskLocal package static var current: WorkCounter?

    private let preparations = Mutex(0)
    private let fftPlans = Mutex(0)
    private let probeSpectra = Mutex(0)
    private let slicePotentials = Mutex(0)
    private let transmissions = Mutex(0)

    package init() {}

    /// Potentials made ready for propagation: transmission functions built (CPU) or a potential prepared on the GPU.
    package var preparationCount: Int { preparations.withLock { $0 } }
    /// FFT plans made for CPU batch workers.
    package var fftPlanCount: Int { fftPlans.withLock { $0 } }
    /// Probe spectra (aperture and aberration phase over the grid) computed for building probes.
    package var probeSpectrumCount: Int { probeSpectra.withLock { $0 } }
    /// Slice potentials built on the CPU (`Potential.build()`), once per distinct slice (#122).
    package var slicePotentialCount: Int { slicePotentials.withLock { $0 } }
    /// Transmission functions computed (CPU) or cached on the GPU, once per distinct slice (#122).
    package var transmissionCount: Int { transmissions.withLock { $0 } }

    package func countPreparation() { preparations.withLock { $0 += 1 } }
    package func countFFTPlan() { fftPlans.withLock { $0 += 1 } }
    package func countProbeSpectrum() { probeSpectra.withLock { $0 += 1 } }
    package func countSlicePotential() { slicePotentials.withLock { $0 += 1 } }
    package func countTransmission() { transmissions.withLock { $0 += 1 } }
}

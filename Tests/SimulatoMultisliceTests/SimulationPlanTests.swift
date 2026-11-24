//
//  SimulationPlanTests.swift
//  Role: the simulation plan (#91): the sampling rule, the grid and abTEM's cutoff angles against inline abTEM 1.0.10
//        pins (tools/pins/simulation-plan/pins.py), the memory table, the steps, the time, and each refusal with a
//        sampling that does fit.
//

import Testing
import SimulatoCore
import SimulatoMultislice

@Suite("Simulation plan")
struct SimulationPlanTests {
    /// abTEM 1.0.10: extent, energy, largest angle → λ, sampling, rounded gpts, `cutoff_angles` on the rounded grid.
    struct Pin {
        let extent: SIMD2<Double>, energy: Double, beta: Double
        let wavelength: Double, sampling: Double, nx: Int, ny: Int, cutoff: (Double, Double)
    }

    static let pins: [Pin] = [
        Pin(extent: [10.86, 15.36], energy: 300e3, beta: 20, wavelength: 0.01968748889772767, sampling: 0.05,
            nx: 225, ny: 320, cutoff: (135.96332111690378, 135.86418119525607)),
        Pin(extent: [20, 20], energy: 200e3, beta: 200, wavelength: 0.025079340317328468, sampling: 0.04179890052888078,
            nx: 480, ny: 480, cutoff: (200.63472253862778, 200.63472253862778)),
        Pin(extent: [25, 31], energy: 80e3, beta: 150, wavelength: 0.04175716056489365, sampling: 0.05,
            nx: 500, ny: 625, cutoff: (278.9378325734896, 277.48306697961584)),
        Pin(extent: [40.1, 40.1], energy: 300e3, beta: 90, wavelength: 0.01968748889772767, sampling: 0.05,
            nx: 810, ny: 810, cutoff: (132.55915217921373, 132.55915217921373)),
    ]

    static let roomy = SimulationLimits(memoryBudget: .max, largestBuffer: .max)

    static func close(_ a: Double, _ b: Double, _ rtol: Double = 1e-12) -> Bool { abs(a - b) <= rtol * abs(b) }

    @Test func theSamplingGridAndCutoffAnglesMatchAbTEM() throws {
        for pin in Self.pins {
            let lambda = try ElectronBeam(energy: pin.energy).wavelength
            #expect(Self.close(lambda, pin.wavelength))
            let sampling = try SimulationPlan.sampling(largestAngle: pin.beta, energy: pin.energy)
            #expect(Self.close(sampling, pin.sampling))
            let plan = try SimulationPlan(.init(extent: pin.extent, thickness: 10, energy: pin.energy, largestAngle: pin.beta),
                                          limits: Self.roomy)
            #expect(plan.grid.nx == pin.nx && plan.grid.ny == pin.ny)
            #expect(plan.grid.extentX == pin.extent.x && plan.grid.extentY == pin.extent.y)
            #expect(Self.close(plan.cutoffAngles.x, pin.cutoff.0) && Self.close(plan.cutoffAngles.y, pin.cutoff.1))
            // The cutoff reaches the largest angle asked for.
            #expect(min(plan.cutoffAngles.x, plan.cutoffAngles.y) >= pin.beta)
        }
    }

    @Test func gridLengthsAreFFTFriendlyAndNoCoarserThanAsked() throws {
        #expect(SimulationPlan.fftFriendlyLength(atLeast: 218) == 225)
        #expect(SimulationPlan.fftFriendlyLength(atLeast: 1) == 1)
        #expect(SimulationPlan.fftFriendlyLength(atLeast: 1024) == 1024)
        // 135 x 123 at 0.1 Å rounds to 135 x 125 on its own (no 1 % match inflating it); a thin axis keeps 2 points.
        let plan = try SimulationPlan(.init(extent: [13.49, 12.30], thickness: 5, energy: 200e3, largestAngle: 30, sampling: 0.1),
                                      limits: Self.roomy)
        #expect(plan.grid.nx == 135 && plan.grid.ny == 125)
        #expect(plan.grid.samplingX <= 0.1 && plan.grid.samplingY <= 0.1)
        let thin = try SimulationPlan(.init(extent: [0.05, 10], thickness: 5, energy: 200e3, largestAngle: 30, sampling: 0.1),
                                      limits: Self.roomy)
        #expect(thin.grid.nx == 2 && thin.grid.ny == 100)
    }

    @Test func memoryStepsAndTimeFollowTheBudgetTable() throws {
        // 128 x 128 (N = 16384), 4 slices of 1 Å, two elements, one plane wave.
        let request = SimulationPlan.Request(extent: [12.8, 12.8], thickness: 4, energy: 200e3, largestAngle: 30,
                                             sampling: 0.1, elements: 2)
        let plan = try SimulationPlan(request, limits: Self.roomy)
        let n = 16384
        #expect(plan.grid.nx == 128 && plan.grid.ny == 128)
        #expect(plan.sliceCount == 4 && plan.batchSize == 1)
        #expect(plan.memory.waves == 40 * n && plan.memory.potential == 104 * n)
        #expect(plan.memory.propagators == 12 * n && plan.memory.scan == 0 && plan.memory.storedExitWaves == 0)
        #expect(plan.memory.total == 156 * n)
        #expect(plan.largestBufferBytes == 8 * n)
        #expect(plan.propagationSteps == 4 && plan.fftCount == 8)
        #expect(try plan.estimatedSeconds(stepsPerSecond: 2) == 2)

        // 100 probes in batches of 10, 3 configurations, 2 stored planes.
        var scan = request
        scan.illumination = .probes(positions: 100)
        scan.configurations = 3
        scan.storedExitPlanes = 2
        scan.batchBytes = 24 * n * 10
        let s = try SimulationPlan(scan, limits: Self.roomy)
        #expect(s.batchSize == 10)
        #expect(s.memory.waves == 400 * n && s.memory.potential == 104 * n && s.memory.propagators == 12 * n)
        #expect(s.memory.scan == 48 * n && s.memory.storedExitWaves == 80 * n)
        #expect(s.largestBufferBytes == 80 * n)

        // Uploaded slices: 4 N each and 16 N of workspace, whatever the elements.
        var uploaded = request
        uploaded.potential = .slices
        let u = try SimulationPlan(uploaded, limits: Self.roomy)
        #expect(u.memory.potential == (4 * 4 + 16) * n && u.memory.total == (40 + 32 + 12) * n)
        #expect(s.propagationSteps == 4 * 100 * 3 && s.fftCount == 2400)
    }

    @Test func eachLimitIsRefusedWithASamplingThatFits() throws {
        let request = SimulationPlan.Request(extent: [12.8, 12.8], thickness: 4, energy: 200e3, largestAngle: 30,
                                             sampling: 0.1, elements: 2)
        let cases: [(SimulationLimits, String)] = [
            (SimulationLimits(memoryBudget: 2_000_000, largestBuffer: .max), "budget"),
            (SimulationLimits(memoryBudget: .max, largestBuffer: 100_000), "buffer"),
            (SimulationLimits(memoryBudget: .max, largestBuffer: .max, largestGridLength: 100), "grid"),
        ]
        for (limits, kind) in cases {
            do {
                _ = try SimulationPlan(request, limits: limits)
                Issue.record("\(kind) not refused")
            } catch let error as SimulationPlanError {
                let fits: Double?
                switch (error, kind) {
                case (.overBudget(let required, let budget, let s), "budget"):
                    #expect(required == 156 * 16384 && budget == 2_000_000)
                    fits = s
                case (.bufferTooLarge(let required, let limit, let s), "buffer"):
                    #expect(required == 8 * 16384 && limit == 100_000)
                    fits = s
                case (.gridTooLarge(let nx, let ny, let limit, let s), "grid"):
                    #expect(nx == 128 && ny == 128 && limit == 100)
                    fits = s
                default:
                    Issue.record("\(kind): wrong refusal \(error)")
                    continue
                }
                let fit = try #require(fits, "\(kind): no sampling fits")
                #expect(fit > 0.1 && fit < 0.2, "\(kind): \(fit)")
                #expect(!error.description.isEmpty)
                var coarser = request
                coarser.sampling = fit
                #expect(throws: Never.self) { try SimulationPlan(coarser, limits: limits) }
            }
        }
    }

    @Test func gridRefusalComesBeforeTheMemoryRefusals() throws {
        let request = SimulationPlan.Request(extent: [12.8, 12.8], thickness: 4, energy: 200e3, largestAngle: 30, sampling: 0.1)
        let tight = SimulationLimits(memoryBudget: 1, largestBuffer: 1, largestGridLength: 100)
        let error = #expect(throws: SimulationPlanError.self) { try SimulationPlan(request, limits: tight) }
        guard case .gridTooLarge = error else {
            Issue.record("expected the grid refusal first, got \(String(describing: error))")
            return
        }
    }

    @Test func aBudgetNoGridMeetsHasNoSampling() throws {
        let request = SimulationPlan.Request(extent: [12.8, 12.8], thickness: 4, energy: 200e3, largestAngle: 30, sampling: 0.1)
        let error = #expect(throws: SimulationPlanError.self) {
            try SimulationPlan(request, limits: SimulationLimits(memoryBudget: 10, largestBuffer: .max))
        }
        guard case .overBudget(_, _, let fits) = error else {
            Issue.record("expected the budget refusal, got \(String(describing: error))")
            return
        }
        #expect(fits == nil)
        #expect(error?.description.contains("no sampling fits") == true)
    }

    @Test func invalidRequestsAreRefused() throws {
        let good = SimulationPlan.Request(extent: [10, 10], thickness: 4, energy: 200e3, largestAngle: 30)
        var bad: [SimulationPlan.Request] = []
        var r = good; r.extent = [0, 10]; bad.append(r)
        r = good; r.extent = [.nan, 10]; bad.append(r)
        r = good; r.thickness = -1; bad.append(r)
        r = good; r.energy = -1; bad.append(r)
        r = good; r.largestAngle = 0; bad.append(r)
        r = good; r.sampling = .infinity; bad.append(r)
        r = good; r.illumination = .probes(positions: 0); bad.append(r)
        r = good; r.configurations = 0; bad.append(r)
        r = good; r.elements = 0; bad.append(r)
        r = good; r.sliceThickness = 0; bad.append(r)
        for request in bad {
            #expect(throws: (any Error).self) { try SimulationPlan(request, limits: Self.roomy) }
        }
        let plan = try SimulationPlan(good, limits: Self.roomy)
        #expect(throws: SimulationPlanError.self) { try plan.estimatedSeconds(stepsPerSecond: 0) }
        #expect(throws: SimulationPlanError.self) { try plan.estimatedSeconds(stepsPerSecond: .nan) }
    }

    @Test func thisMacHasPositiveLimits() {
        let limits = SimulationLimits.thisMac
        #expect(limits.memoryBudget > 0 && limits.largestBuffer > 0 && limits.largestGridLength == 4096)
    }
}

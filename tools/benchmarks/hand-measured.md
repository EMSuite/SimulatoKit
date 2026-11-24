## HRTEM image series (measured by hand)

`ContrastTransferFunction.images(of:defocus:backend:)`: twenty defocus values (−100 to 90 Å), 20 mrad CTF, Al [001] at
200 keV, exit waves from a CPU multislice. Not run by `run.sh`: measured on 24 Nov 2026 (Apple M5 Pro, Xcode 27, SimulatoKit
main 2e58364, load about 1.5) with `SIMULATO_BENCHMARK=1 swift test -c release --filter HRTEMSeriesBenchmarkTests` for the
first row and a throwaway test for the rest (#89). Median in ms of 7 runs after a warm-up (3 on the CPU at 1024²); no
abTEM column.

| case | images | kit CPU f64 | kit Metal f32 | CPU / Metal | max abs(CPU − Metal) / max |
|---|---|---|---|---|---|
| one 100 × 150 unit, one exit wave | 20 | 11.1 (10.9-11.4) | 11.8 (11.7-11.9) | 0.94x | 5.5e-7 |
| 256², 5 thicknesses | 120 | 108 | 75.1 | 1.44x | 7.3e-7 |
| 512², 5 thicknesses | 120 | 572 | 297 | 1.93x | 9.0e-7 |
| 1024², 5 thicknesses | 120 | 2376 | 1073 | 2.21x | 1.0e-6 |

Decision: CPU at one small unit (the GPU only ties it), Metal from 256² (break-even between 100 × 150 and 256²).

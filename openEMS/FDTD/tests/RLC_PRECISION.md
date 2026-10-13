# RLC auxiliary-state precision regression

Small timesteps can expose an instability in the single-precision lumped RLC
recurrence. The operator computes coefficients in double but previously stored
them in `FDTD_FLOAT`; both CPU and CUDA also stored the voltage/current history
in that type. This change keeps the RLC coefficients and auxiliary state in
double, including CUDA allocation sizes and kernels. The domain E/H arrays and
ordinary field updates retain their existing precision.

## Reproducer

`fixtures/rlc_precision.xml` is a synthetic 17 x 13 x 13 PEC model. A Gaussian
950 MHz / 450 MHz excitation drives a three-cell series RLC branch
(R = 0.1662729 ohm, L = 0.3750881 nH, C = 10 microfarad), coupled through PEC
rails to a 100 pF load. The fixed timestep is approximately 29.5736 fs.
The model is supplied as XML; Python bindings and NumPy are not required.

Run from the repository root with a CUDA-enabled build and an unused output
directory:

```sh
CUDA_VISIBLE_DEVICES=0 python3 openEMS/FDTD/tests/run_rlc_precision.py \
  build/rlc-precision-run --binary /absolute/path/to/openEMS
```

The runner uses one CPU thread, runs `basic` then `cuda`, and selects the first
visible CUDA device. Use `--engine basic` for a CPU-only check. It inherits
`LD_LIBRARY_PATH`; ensure that it resolves the intended build's `libopenEMS`,
rather than an older installed copy. Each subprocess has a five-minute timeout.
For example, with dependencies already installed in `$OPENEMS_DEPS`:

```sh
cmake -S openEMS -B build/rlc-double \
  -DCMAKE_BUILD_TYPE=Release -DOPENEMS_ENABLE_CUDA=ON \
  -DFPARSER_ROOT_DIR="$OPENEMS_DEPS" -DCSXCAD_ROOT_DIR="$OPENEMS_DEPS" \
  -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/rlc-double --target openEMS_bin --parallel 4
LD_LIBRARY_PATH="$PWD/build/rlc-double:$OPENEMS_DEPS/lib:${LD_LIBRARY_PATH:-}" \
  CUDA_VISIBLE_DEVICES=0 python3 openEMS/FDTD/tests/run_rlc_precision.py \
  build/rlc-precision-run --binary build/rlc-double/openEMS
```

Set `CUDA_ROOT` to an existing compatible toolkit and adjust the architecture
for the target GPU. These commands do not install or change system defaults.
Rebuild consumers of the RLC extension headers together with the library:
the coefficient and state member types have changed.

The fixed observation is 786,432 steps (23.26 ns). Both probe files must contain
finite, increasing timestamps and cover the observation window. Peak voltage
must be between 1 microvolt and 1 millivolt, and the last tenth's peak must be
below 25% of the full peak. These are fixture-specific stability guards, not
physical accuracy criteria. The CPU/CUDA comparison additionally requires
maximum error <= 0.2% and RMS error <= 0.05%, normalized by the CPU peak.
The shorter existing regression's `1e-4` tolerance is unchanged.

Output includes `solver.log`, both probes, and `rlc_precision_report.json`.
The command returns nonzero on failure, including incomplete iterations or
unexcited output. Existing logs/probes can be checked without starting a solver:

```sh
python3 openEMS/FDTD/tests/run_rlc_precision.py build/rlc-precision-run --check-existing
```

## Recorded validation

The five changed engine files were built on Linux with CUDA 12.6 and tested on
an RTX 4090, against base commit `568b4454e4b4ee06beb671eb72bfdc1db67c28ca`.
At 262,144 steps, the same XML gave the following load-voltage peaks:

| Engine | Original float RLC | Double RLC coefficients/state |
| --- | ---: | ---: |
| CPU basic | 1.93283e15 V | 1.19702e-4 V |
| CUDA | 1.86236e15 V | 1.19717e-4 V |

The double runs completed 786,432 steps with finite, decaying waveforms.
Maximum/RMS normalized errors were 0.122154% / 0.027915% for load voltage and
0.122619% / 0.035961% for field voltage. The checked-in runner was also replayed
against these archived outputs; both passed. Original float outputs were
rejected for excessive peak voltage.

The published engine code matches those tested sources, apart from explanatory
comments. The fixture differs from the tested XML only in two property names.
The existing Metal changes are retained; Metal was not numerically tested by
this validation. These results establish a stability improvement for this
reproducer, not general RF accuracy, convergence, or a fix for all boundary
instabilities. Coefficient rounding and state rounding were changed together;
their individual contributions were not separated.

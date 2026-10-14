# openEMS CUDA + Metal

This repository is an experimental GPU-accelerated fork of
[openEMS](https://github.com/thliebig/openEMS), a free and open-source 3D
finite-difference time-domain (FDTD) electromagnetic solver. It adds optional
NVIDIA CUDA and Apple Metal time-stepping backends while preserving the
original CPU engines as the default.

> **Project status:** research software. CUDA has been validated on selected
> Cartesian models and rejects unsupported physics during setup. The macOS
> Metal backend is an **experimental hybrid** implementation: backend source
> compilation and a Metal shader/compute-pipeline smoke test have passed, but a
> complete solver build and CPU-vs-Metal numerical simulation have **not** yet
> been validated. Neither GPU backend implements every openEMS feature. This
> fork is not an official openEMS release; see the
> [upstream project](https://github.com/thliebig/openEMS-Project) for the
> maintained CPU implementation and general documentation.

The [Linux CUDA 12.6 validation](#linux-cuda-126-validation) below records the
October 2026 RLC instability investigation, corrective patch, reproduction
commands, and subsequent large-model runs.

## What changed from upstream

The fork is based on `openEMS-Project` commit
`84703e254dafe84ed2f06056e5f105067b8f1910` and `openEMS` commit
`27dcde1c24699bd1f3d2cfa67bfad7883cb5907e`, both dated August 2026. The
principal changes are:

| Area | Change |
| --- | --- |
| CUDA engine | Added a single-GPU, single-precision Cartesian Yee-grid engine in `openEMS/FDTD/engine_cuda.cu` with GPU-resident E/H fields and update coefficients. |
| CUDA operator | Added `Operator_CUDA`, device selection, memory setup, extension validation, and a pinned host mirror for existing processing code. |
| Metal engine (experimental) | Added `Engine_Metal` and `Operator_Metal`: Cartesian E/H field updates use Apple Metal compute shaders, while existing CPU extension hooks continue to run on the host. |
| Build system | Added opt-in `OPENEMS_ENABLE_CUDA` and `OPENEMS_ENABLE_METAL` CMake options. The CUDA architecture setting is forwarded to the solver as before; CPU-only builds remain available. |
| Runtime selection | Added `--engine=cuda`, `--cuda-device=<n>`, `--cuda-debug-sync`, and `--engine=metal`. Python `Run()` keyword arguments map to the same engine options. |
| CUDA excitation and boundaries | Ported voltage/current excitation, TFSF plane waves, UPML, PEC/PMC boundaries, supported lumped RLC elements, and periodic steady-state detection. TFSF includes fractional delay and numerical phase-velocity compensation. |
| CUDA processing | Added explicit device-to-host synchronization only at requested processing timesteps so existing probes, dumps, NF2FF, and SAR processing can use the normal engine interface. |
| CUDA frequency-domain dumps | Added bounded sample windows, explicit sampling interval, periodic complex-amplitude averaging, double-precision phase/accumulation, and HDF5 metadata. |
| CUDA error handling | Added clearer failures for invalid engine options, CUDA setup/runtime errors, and physics that has not been ported. Python propagates C++ runtime exceptions. |
| Validation | Added a numerical phase-velocity unit test and deterministic CPU/CUDA regression generators and comparisons under `openEMS/FDTD/tests/`, plus Metal smoke tests and a CPU/Metal comparison runner in `scripts/verification/`. |
| Windows Python loading | Added explicit openEMS and CUDA DLL-directory handling so a CUDA build can coexist with a separate CSXCAD/dependency installation. |

The original component repositories are vendored as ordinary directories in
this repository rather than Git submodules. This makes a single clone a
complete source snapshot; syncing future upstream changes is therefore a
manual operation.

## Supported CUDA model features

The current backend supports:

- Cartesian graded Yee meshes, including unequal and nonuniform X/Y/Z spacing.
- Isotropic permittivity, permeability, and conductivity represented by the
  base openEMS update coefficients.
- PEC and PMC boundaries and six-sided UPML.
- Voltage and current excitation, including per-cell delay and periodic
  signals.
- TFSF plane-wave excitation, including oblique incidence, fractional delays,
  and numerical phase-velocity compensation.
- Parallel resistive/RC loads and dynamic series/parallel lumped RLC elements.
- Periodic-excitation steady-state detection.
- Existing probes, field dumps, NF2FF, SAR, and other processing that consumes
  the standard CPU engine interface.

CPU and CUDA lumped RLC coefficients and auxiliary history use double precision
to avoid a reproduced small-timestep instability. Domain fields retain their
existing precision. See the [RLC precision regression](openEMS/FDTD/tests/RLC_PRECISION.md)
for the reproducer, measured results, and validation limits.

The following active features are deliberately rejected by the CUDA engine:

- Cylindrical coordinates and cylindrical multigrid.
- Mur/local absorbing boundaries.
- Lorentz, Drude, and Debye dispersive material updates.
- Conducting-sheet and absorbing-sheet extensions.

Select `--engine=multithreaded` for a model that uses one of these features.

## Experimental Apple Metal backend (macOS)

Build with `OPENEMS_ENABLE_METAL=ON` and select `--engine=metal`.
The CUDA and existing CPU backends are unchanged. Metal targets macOS
(primarily Apple Silicon); it does **not** run on Linux or Windows.

**What is implemented:**

- Cartesian Yee-grid electric-field and magnetic-field updates execute on the
  GPU using the two Metal compute kernels in `openEMS/FDTD/engine_metal.mm`.
- The existing CPU-side openEMS extension pipeline is called around each
  field update, including excitation, boundaries, and material/lumped-element
  extensions. These paths have **not** been validated for numerical parity
  with the CPU solver.
- The original CPU-side processing interface remains available for probes
  and other output processing. Cylindrical-grid models are rejected.

**Important limitations:** This is a correctness-oriented **CPU/Metal hybrid**,
not a feature-complete port of the CUDA GPU kernels. Extension calculations
still execute on the CPU. Field arrays are copied between the host representation
and Metal buffers and synchronized for each voltage/current update, so it can
be **slower than the CPU engine**, especially for small models. Numerical
accuracy, supported physics, and end-to-end speedup remain unverified. Use a
CPU engine as the reference for production or untested models.

For implementation details and the verification commands, see
[`scripts/verification/README.md`](scripts/verification/README.md).

## Requirements

**Common dependencies:** CSXCAD, fparser, TinyXML, HDF5, VTK, Boost, CMake,
and a suitable C++ compiler.

**For CUDA:** An NVIDIA GPU with compatible drivers and CUDA Toolkit,
CMake 3.18 or newer, and a compiler supported by the selected CUDA Toolkit.
Windows development was validated with Visual Studio 2022, Ninja, and vcpkg;
see [`openEMS/CUDA_WINDOWS.md`](openEMS/CUDA_WINDOWS.md). Validate Linux/CUDA
builds on the intended driver/toolkit before running large simulations.

**For Metal:** macOS with a Metal-capable GPU (Apple Silicon is the primary
target), AppleClang/Xcode Command Line Tools with the Metal SDK, and the common
openEMS dependencies. CMake 3.21 or newer is recommended and required for the
standalone smoke-test project. An NVIDIA GPU or CUDA Toolkit is **not**
required for Metal.

## Build

The top-level project builds the bundled fparser, CSXCAD, and openEMS sources.
The GUI can be disabled for headless or container use:

```bash
cmake -S . -B build/cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/install/cuda" \
  -DBUILD_APPCSXCAD=NO \
  -DOPENEMS_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89

cmake --build build/cuda --parallel
```

Replace `89` with the compute capability of the target GPU, or use a
semicolon-separated architecture list for a portable binary. A CPU-only build
can omit `OPENEMS_ENABLE_CUDA` and `CMAKE_CUDA_ARCHITECTURES`.

If CSXCAD and fparser are already installed, the `openEMS/` subproject can be
configured directly. The detailed Windows command in
[`openEMS/CUDA_WINDOWS.md`](openEMS/CUDA_WINDOWS.md) uses this workflow and also
enables the focused C++ tests with `-DOPENEMS_BUILD_TESTS=ON`.

### macOS / Apple Metal (experimental)

Install the common openEMS dependencies first. For a headless macOS build from
the repository root:

```bash
cmake -S . -B build/metal \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/install/metal" \
  -DBUILD_APPCSXCAD=NO \
  -DOPENEMS_ENABLE_CUDA=OFF \
  -DOPENEMS_ENABLE_METAL=ON

cmake --build build/metal --parallel
```

This top-level build also builds the bundled `fparser` and `CSXCAD` projects.
The solver requires additional system libraries (including TinyXML, HDF5,
VTK, and Boost). Configuration will fail until the dependencies are installed.
Alternatively, point `FPARSER_ROOT_DIR` and `CSXCAD_ROOT_DIR` at an existing
installation and configure the `openEMS/` subproject directly. A complete
macOS solver build has **not** yet been verified for this fork.

## Run

The standalone solver uses the upstream CPU engine by default; select a GPU
backend explicitly when its build option is enabled:

```bash
openEMS model.xml --engine=cuda --cuda-device=0
```

For a macOS build configured with `OPENEMS_ENABLE_METAL=ON`:

```bash
openEMS model.xml --engine=metal
```

`--cuda-debug-sync` synchronizes after every timestep to surface kernel errors
near their source. It substantially reduces performance and should not be used
for benchmarks or production runs.

The Python wrapper forwards keyword arguments to the same option parser,
replacing underscores with hyphens:

```python
fdtd.Run(
    simulation_directory,
    engine="cuda",
    cuda_device=0,
    cuda_debug_sync=False,
)
```

With the experimental Metal build, pass `engine="metal"` instead:

```python
fdtd.Run(simulation_directory, engine="metal")
```

On Windows, set `OPENEMS_INSTALL_PATH` to the CUDA build's `bin` directory and
ensure the CSXCAD and CUDA runtime DLL directories are available before
importing `openEMS`.

## Periodic frequency-domain output

For continuous-wave datasets, frequency-domain dump types 10 through 15 can
accumulate the requested complex field on the fly and write only the final
result. Three optional custom dump attributes were added:

- `FDInterval`: timestep interval between accumulated samples.
- `FDFirstSample` and `FDLastSample`: one-based inclusive sample window.
- `FDNormalization=periodic-average`: return the mean periodic complex
  amplitude instead of the legacy single-sided time integral.

The HDF5 output records the sample count, interval, window, and normalization.
Complex phase and accumulation use double precision; the underlying FDTD
fields remain single precision.

## Validation

### CUDA tests

Build the focused test and run it through CTest:

```bash
cmake -S openEMS -B build/openems-cuda \
  -DOPENEMS_ENABLE_CUDA=ON \
  -DOPENEMS_BUILD_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DFPARSER_ROOT_DIR=/path/to/prefix \
  -DCSXCAD_ROOT_DIR=/path/to/prefix
cmake --build build/openems-cuda --parallel
ctest --test-dir build/openems-cuda --output-on-failure
```

The deterministic CPU/CUDA regression runner is:

```bash
python openEMS/FDTD/tests/run_cuda_regression.py \
  build/openems-cuda/openEMS \
  build/cuda-regression \
  --cuda-debug-sync
```

The regression cases cover anisotropic meshes, base excitation, UPML, a
parallel resistor, a series RLC load, sinusoidal TFSF excitation, and
steady-state detection. On the original development system, probe timestamps
matched exactly and the largest observed probe error was approximately
`1.34e-5` relative to the CPU peak.

For the longer RLC stability reproducer (23.26 ns, supplied XML):

```bash
CUDA_VISIBLE_DEVICES=0 python3 openEMS/FDTD/tests/run_rlc_precision.py \
  build/rlc-precision-run --binary build/openems-cuda/openEMS
```

This fixture has separate stability and waveform-error limits, described in
[`RLC_PRECISION.md`](openEMS/FDTD/tests/RLC_PRECISION.md).

### Metal smoke test (completed on macOS)

The following smoke test compiles the Metal backend's C++/Objective-C++
translation units, compiles its embedded Metal shaders on a Metal device, and
creates both GPU compute pipelines:

```bash
cmake -S scripts/verification -B build/metal-compile-smoke
cmake --build build/metal-compile-smoke --parallel
cmake --build build/metal-compile-smoke --target test
```

**Observed result (October 2026):** AppleClang 17 backend compilation passed,
and the Metal shader/pipeline runtime smoke test passed (1/1). This is
**not a physical FDTD simulation test** and does not establish numerical parity
or a performance benefit.

### CPU versus Metal numerical regression (not yet completed)

After successfully building the solver and installing the Python `openEMS`,
`CSXCAD`, and NumPy dependencies:

```bash
python3 scripts/verification/verify_metal.py \
  --binary "$PWD/install/metal/bin/openEMS" \
  --work build/metal-regression
```

The runner generates deterministic cases for basic Cartesian FDTD, UPML,
a resistor, series RLC, and sinusoidal TFSF. It compares CPU `basic` and
`metal` probe timestamps and relative waveform error (default tolerance:
`1e-4`). This comparison was prepared but **has not been executed
successfully** because the complete solver dependencies were unavailable.
Additional physics and performance benchmarks are still required.

### CUDA benchmark

On an RTX 4090 with CUDA 13.3, a 129 x 129 x 129 anisotropic mesh run for 512
timesteps measured about 3013 MCells/s (0.36 s) on CUDA and 759 MCells/s (1.45
s) with the multithreaded compressed-SSE engine at its measured optimum of 16
CPU workers. This roughly 4x result covers time stepping with ordinary probe
processing; it excludes geometry/operator construction and should not be
treated as a universal end-to-end speedup. Frequent or large output can become
limited by PCIe transfers and CPU-side processing.

Base CUDA device storage is approximately 72 bytes per Yee cell: six field
components and twelve update-coefficient components stored as `float`. UPML,
dynamic lumped elements, CPU-side operator data, and the host processing mirror
require additional memory.

### Linux CUDA 12.6 validation

Tests conducted on **2026-10-13–14 (KST)** reproduced unphysical late-time
amplification in both CPU and CUDA RLC updates. Keeping the RLC coefficients
and history in double precision removed this growth in the tested cases.
The corrective engine code and public synthetic reproducer are in
[`codex/rlc-double-stability`](https://github.com/present1001/openems-cuda/tree/codex/rlc-double-stability),
commit [`6bd4d0e`](https://github.com/present1001/openems-cuda/commit/6bd4d0e5f561ae925fb12c011d0be6f3d748bbbf).
The engine patch is now **merged into `main`**. The pinned revision below
retains the exact public reproducer used for the recorded CPU/CUDA results.

#### Recorded environment

| Item | Tested configuration |
| --- | --- |
| OS / kernel | Ubuntu 20.04.6 LTS, x86_64; Linux 5.15.0-139-generic |
| CPU / RAM | AMD Ryzen Threadripper PRO 5975WX, 32 cores / 64 logical CPUs; approximately 251.5 GiB OS-visible RAM |
| GPUs | Four NVIDIA GeForce RTX 4090 cards, 24,564 MiB reported per card |
| NVIDIA driver | 580.126.09 |
| CUDA compiler | Toolkit 12.6, `nvcc` 12.6.68; existing `/usr/local/cuda-12.6/bin/nvcc` |
| Build | GCC 9.4.0, CMake 3.31.6, Release, CUDA architecture `89` |
| Python | System Python 3.8.10; application environment Python 3.11.17; the public RLC runner needs only Python 3.8+ and its standard library |
| Resource policy | Application jobs shared a CPU-time limit of 80% of 64 logical CPUs and a RAM limit of 80% of available memory sampled at launch; the production pair used 25 CPU workers per job |

The existing system CUDA and driver installations were retained. An `nvcc`
missing from a noninteractive SSH `PATH` does not imply that the toolkit is
absent: the compiler path above was recorded in the build cache. The CUDA
version displayed by `nvidia-smi` is not the compiler version used for a build.
These are separate measurements from the CUDA 13.3 benchmark above.

#### Amplification case, diagnosis, and changes

One initial large-model CUDA run developed nonfinite energy at timestep
674,523. Shorter reproductions could finish with exit code zero and finite
probes while already growing unphysically. Completion status alone was
therefore insufficient to qualify a result.

The controlled CPU reference was the **same fork's `basic` engine**, not a
different upstream CPU installation. Tests compared identical XML inputs,
examined excitation, probe timestamps and amplitudes, and then changed RLC
precision while preserving the model and timestep. Boundary conditions were
tested separately. Reducing the timestep alone did not resolve the observed
instability; some trials grew faster.
Unexcited trials and Dirac trials with uninitialized excitation parameters
were excluded. Unsupported CUDA boundary options were setup failures, not
evidence of numerical divergence.

| Changed code in `6bd4d0e` | Correction |
| --- | --- |
| `operator_ext_lumpedRLC.h/.cpp` | Store series/parallel RLC coefficients in `double`; retain double precision in the grid-capacitance temporary used during construction. |
| `engine_ext_lumpedRLC.h/.cpp` | Store inductor current and the three-level voltage/current history in `double`, including allocations and history-buffer rotation. |
| `engine_cuda.cu` | Use double RLC kernel arguments, arithmetic and device buffers; update typed allocations/copies, zero-initialization byte counts and buffer-rotation pointers consistently. |
| `FDTD/tests/run_rlc_precision.py` and synthetic XML fixture | Check completed steps, finite and excited probes, timestamp coverage, late-time decay and CPU/CUDA waveform error; return nonzero when a check fails. |

The ordinary E/H arrays and field-update coefficients remain single precision.
The tests changed RLC coefficient and state precision together, so they do not
separate the contribution of each. A separate small-domain absorbing-boundary
instability was observed even without the suspect RLC branch; that issue was
not fixed or explained by this patch. The result is a demonstrated RLC
stability correction, not a universal cure for boundary or discretization errors.

#### Public, reproducible CPU/CUDA test

The synthetic fixture uses **17 × 13 × 13 = 2,873 cells**, PEC boundaries,
Gaussian excitation, and a timestep of approximately **29.5736 fs**. No
application PCB or geometry file is needed. At 262,144 steps (7.75 ns), the
recorded maximum absolute load voltages were:

| Engine | Original float RLC | Double RLC coefficients/history |
| --- | ---: | ---: |
| CPU `basic` | 1.93283e15 V | 1.19702e-4 V |
| CUDA | 1.86236e15 V | 1.19717e-4 V |

Both patched engines also completed **786,432 steps (23.26 ns)** with finite,
decaying probes. Error is normalized by the CPU waveform's peak:

| Probe | Maximum error | RMS error |
| --- | ---: | ---: |
| Load voltage | 0.122154% | 0.027915% |
| Field voltage | 0.122619% | 0.035961% |

The runner requires maximum/RMS error at most 0.2% / 0.05%, a peak between
1 microvolt and 1 millivolt, and a final-tenth peak below 25% of the full peak.
These amplitude/decay limits apply to this fixture only. Rechecking archived
patched output passed; original float output was rejected. Full fixture and
validation details are in
[`RLC_PRECISION.md`](https://github.com/present1001/openems-cuda/blob/6bd4d0e5f561ae925fb12c011d0be6f3d748bbbf/openEMS/FDTD/tests/RLC_PRECISION.md).

To reproduce, use a fresh clone and existing compatible dependencies. Set
`OPENEMS_DEPS` to the prefix containing CSXCAD and fparser, and make a suitable
CMake available in this shell. This example selects the installed toolkit
explicitly without changing system defaults:

```bash
git clone --branch codex/rlc-double-stability \
  https://github.com/present1001/openems-cuda.git openems-rlc-validation
cd openems-rlc-validation
git checkout --detach 6bd4d0e5f561ae925fb12c011d0be6f3d748bbbf
export CUDA_ROOT=/usr/local/cuda-12.6
export OPENEMS_DEPS=/path/to/existing/dependency-prefix

cmake -S openEMS -B build/rlc-double \
  -DCMAKE_BUILD_TYPE=Release -DOPENEMS_ENABLE_CUDA=ON \
  -DFPARSER_ROOT_DIR="$OPENEMS_DEPS" -DCSXCAD_ROOT_DIR="$OPENEMS_DEPS" \
  -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/rlc-double --target openEMS_bin --parallel 4
export LD_LIBRARY_PATH="$PWD/build/rlc-double:$OPENEMS_DEPS/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ldd build/rlc-double/openEMS | grep libopenEMS
CUDA_VISIBLE_DEVICES=0 python3 openEMS/FDTD/tests/run_rlc_precision.py \
  build/rlc-check --binary build/rlc-double/openEMS
```

Verify that `ldd` resolves this build's `libopenEMS`, not an older installation.
Use a new output directory for a new run. To recheck saved output without
restarting the solver, use the same runner with `build/rlc-check --check-existing`.
For a float control, build baseline commit `568b4454e4b4ee06beb671eb72bfdc1db67c28ca`
separately and pass that binary to the patched runner with the same fixture;
its stability checks are expected to reject the unstable output.

#### Original-grid boundary control and longer runs

A private application model was then checked without simplifying its
**365 × 501 × 71 = 12,983,415-cell** grid. The bounded CUDA comparison used
271,620 steps (8.03278 ns). Grid, materials, geometry and excitation were held
fixed; the PEC control replaced only the six `PML_8` boundary settings.
The float/PML row reused the matching interval of the archived original run;
the other three cases were newly executed. Probe sample times matched.

| RLC precision / boundary | Final logged energy | Fitted energy multiplier per 10,000 steps after excitation |
| --- | ---: | ---: |
| Float / PML_8 | 1.28e-3 | 7.186 (growth) |
| Float / PEC | 4.40e-3 | 7.251 (growth) |
| Double / PML_8 | 1.05e-15 | 0.597 (decay) |
| Double / PEC | 2.95e-14 | 0.900 (decay) |

Growth persisted with PEC, so PML was not necessary for this failure.
The corrected engine decayed with the original PML_8. This bounded test reached
its step limit; it did not establish long-time or spatial convergence.

Two longer cases subsequently ran concurrently on GPU 0 and GPU 1 with the
corrected engine, unchanged grid and six-sided PML_8. Both used a maximum of
2,000,000 steps and an energy stopping threshold of `1e-7` (-70 dB).
Case labels below are anonymized:

| Case | Final steps | Simulated time | Final energy (dB) | Solver time | Total job time |
| --- | ---: | ---: | ---: | ---: | ---: |
| P1 / GPU 0 | 1,662,918 | 49.17847 ns | -70.10546 | 6,276.52 s | 8,168.52 s |
| P2 / GPU 1 | 510,042 | 15.08379 ns | -70.09948 | 1,925.15 s | 2,943.64 s |

Both reached the requested energy criterion with finite probes and
postprocessed output. P1 passed the application's temporal plateau checks;
P2 did not meet its 20 ns minimum observation requirement. **Neither case
established spatial mesh convergence.** The pair's elapsed time was about
2 h 16 min; total job time includes setup and postprocessing, unlike solver
time. No matched CPU production run was completed, so these timings do not
establish a CPU-to-GPU speedup.

#### Multi-GPU execution and evidence scope

This backend uses **one GPU per solver process**. Multiple GPUs run independent
models; a single model is not split across four cards. GPU 0/1 routing was
checked against the small patched reference, with identical sampled values.
The bounded boundary comparison also used GPU 2; GPU 3 was available but was
not used for the reported production pair.

For separate jobs with already-generated XML, the following is a launch
pattern, not a shipped application test. Run each model from a separate output
directory. `CUDA_VISIBLE_DEVICES` maps the selected physical GPU to logical
device zero inside that process:

```bash
solver="$PWD/build/rlc-double/openEMS"
# Place independent inputs at jobs/case0/model.xml and jobs/case1/model.xml.
(cd jobs/case0 && CUDA_VISIBLE_DEVICES=0 "$solver" model.xml \
  --engine=cuda --cuda-device=0 --numThreads=25 > solver.log 2>&1) & p0=$!
(cd jobs/case1 && CUDA_VISIBLE_DEVICES=1 "$solver" model.xml \
  --engine=cuda --cuda-device=0 --numThreads=25 > solver.log 2>&1) & p1=$!
rc=0
wait "$p0" || rc=1
wait "$p1" || rc=1
test "$rc" -eq 0
```

This example does not implement the server's aggregate CPU/RAM limits or the
application's convergence checks. Those orchestration tools remain external
to this repository. Check energy, finite probes and observation coverage even
when both processes exit zero; use the synthetic runner for the public test.

Only aggregate measurements from the private models are published here.
PCB/schematic files, extracted meshes or model XML, layout images, private
logs, raw RF responses, archives, server addresses and account details are
excluded. The synthetic RLC fixture is the public reproducer; private-model
results are reported observations and cannot be independently regenerated
from this repository alone. Metal was not evaluated by these Linux tests.

## Repository layout and provenance

| Directory | Component | Vendored upstream commit |
| --- | --- | --- |
| `openEMS/` | FDTD solver, CUDA and Metal modifications | `27dcde1c24699bd1f3d2cfa67bfad7883cb5907e` |
| `CSXCAD/` | Continuous-structure geometry library | `cb931b7475049e9292bf3ca67631041cc8bd8eb6` |
| `fparser/` | Function parser | `4b9c845b449b520c4b8c5f23c74cd04820084f81` |
| `QCSXCAD/` | Qt geometry viewer library | `acf34f4ddad3a3749ee6e5b58339f63a57318072` |
| `AppCSXCAD/` | Geometry viewer application | `6dc4c0b398de10c5a8b844a57280a81d5f87e79f` |
| `hyp2mat/` | HyperLynx conversion utility | `b15f749849610066e6c09da5b5e3f97e3317ba8a` |
| `CTB/` | Circuit toolbox | `5ac29367ebb7e30e78dd3db13a21464756855599` |

General openEMS usage and tutorials are available at
[docs.openEMS.de](https://docs.openems.de). The upstream project website is
[openEMS.de](https://openems.de).

## License and attribution

The vendored components retain their original copyright notices and license
files, including `openEMS/COPYING`, `CSXCAD/COPYING`, `QCSXCAD/COPYING`, and
`AppCSXCAD/COPYING`. The CUDA and Metal modifications are derivative work and are
distributed under the applicable license of the modified component. Please
preserve upstream attribution when redistributing this fork.

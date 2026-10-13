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

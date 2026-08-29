# openEMS CUDA

This repository is an experimental CUDA-enabled fork of
[openEMS](https://github.com/thliebig/openEMS), a free and open-source 3D
finite-difference time-domain (FDTD) electromagnetic solver. It adds an opt-in
NVIDIA CUDA time-stepping engine while retaining the original CPU engines as
the default.

> **Project status:** research software. The CUDA backend has been validated on
> selected Cartesian models, but it does not yet implement every openEMS
> feature. Unsupported physics is rejected during setup instead of silently
> producing an incomplete simulation. This fork is not an official openEMS
> release; use the [upstream project](https://github.com/thliebig/openEMS-Project)
> for the fully supported CPU implementation and general documentation.

## What changed from upstream

The fork is based on `openEMS-Project` commit
`84703e254dafe84ed2f06056e5f105067b8f1910` and `openEMS` commit
`27dcde1c24699bd1f3d2cfa67bfad7883cb5907e`, both dated August 2026. The
principal changes are:

| Area | Change |
| --- | --- |
| CUDA engine | Added a single-GPU, single-precision Cartesian Yee-grid engine in `openEMS/FDTD/engine_cuda.cu` with GPU-resident E/H fields and update coefficients. |
| CUDA operator | Added `Operator_CUDA`, device selection, memory setup, extension validation, and a pinned host mirror for existing processing code. |
| Build system | Added the opt-in `OPENEMS_ENABLE_CUDA` CMake option and propagation of `CMAKE_CUDA_ARCHITECTURES` through the top-level build. CPU-only builds are unchanged. |
| Runtime selection | Added `--engine=cuda`, `--cuda-device=<n>`, and the diagnostic `--cuda-debug-sync` command-line options. Python `Run()` keyword arguments map to the same options. |
| Excitation and boundaries | Ported voltage/current excitation, TFSF plane waves, UPML, PEC/PMC boundaries, supported lumped RLC elements, and periodic steady-state detection. TFSF includes fractional delay and numerical phase-velocity compensation. |
| Processing | Added explicit device-to-host synchronization only at requested processing timesteps so existing probes, dumps, NF2FF, and SAR processing can use the normal engine interface. |
| Frequency-domain dumps | Added bounded sample windows, explicit sampling interval, periodic complex-amplitude averaging, double-precision phase/accumulation, and HDF5 metadata. |
| Error handling | Added clearer failures for invalid engine options, CUDA setup/runtime errors, and physics that has not been ported. Python propagates C++ runtime exceptions. |
| Validation | Added a numerical phase-velocity unit test and deterministic CPU/CUDA regression generators and comparisons under `openEMS/FDTD/tests/`. |
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

The following active features are deliberately rejected by the CUDA engine:

- Cylindrical coordinates and cylindrical multigrid.
- Mur/local absorbing boundaries.
- Lorentz, Drude, and Debye dispersive material updates.
- Conducting-sheet and absorbing-sheet extensions.

Select `--engine=multithreaded` for a model that uses one of these features.

## Requirements

- A CUDA-capable NVIDIA GPU and compatible display/compute driver.
- NVIDIA CUDA Toolkit.
- CMake 3.18 or newer for CUDA builds.
- A C++ compiler supported by both openEMS and the selected CUDA Toolkit.
- The normal openEMS dependencies: CSXCAD, fparser, TinyXML, HDF5, VTK, and
  Boost.

Windows development has been validated with Visual Studio 2022, Ninja, and
vcpkg. See [`openEMS/CUDA_WINDOWS.md`](openEMS/CUDA_WINDOWS.md) for the complete
tested Windows setup. Linux/CUDA container builds should be validated on the
target driver and CUDA versions before running large simulation campaigns.

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

## Run

The standalone solver continues to use the upstream CPU engine unless CUDA is
requested explicitly:

```bash
openEMS model.xml --engine=cuda --cuda-device=0
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
| `openEMS/` | FDTD solver and CUDA modifications | `27dcde1c24699bd1f3d2cfa67bfad7883cb5907e` |
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
`AppCSXCAD/COPYING`. The CUDA modifications are derivative work and are
distributed under the applicable license of the modified component. Please
preserve upstream attribution when redistributing this fork.

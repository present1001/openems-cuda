# Native CUDA engine on Windows

openEMS can optionally build a native NVIDIA CUDA time-stepping engine on
64-bit Windows. The CPU engines remain the default and a CPU-only build is
unchanged.

## Requirements

- Windows 10 or 11, x64
- Visual Studio 2022 with the **Desktop development with C++** workload
- CMake 3.18 or newer and Ninja
- NVIDIA display driver and CUDA Toolkit
- The normal openEMS dependencies (CSXCAD, fparser, TinyXML, HDF5, VTK, and
  Boost), either installed by the openEMS-Project build or provided through
  vcpkg

The CUDA Toolkit version must support the installed Visual C++ toolset and the
target GPU. The GeForce RTX 4090 uses compute capability 8.9, so its native
CMake architecture value is `89`.

## Configure and build

Run the following from an x64 Visual Studio Developer PowerShell. Adjust the
dependency and install paths for the local machine:

```powershell
cmake -S openEMS -B build\windows-x64-cuda\openEMS -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_INSTALL_PREFIX="$PWD\install\windows-x64-cuda" `
  -DFPARSER_ROOT_DIR="$PWD\install\windows-x64" `
  -DCSXCAD_ROOT_DIR="$PWD\install\windows-x64" `
  -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake `
  -DVCPKG_TARGET_TRIPLET=x64-windows-release `
  -DOPENEMS_ENABLE_CUDA=ON `
  -DCMAKE_CUDA_ARCHITECTURES=89 `
  -DOPENEMS_BUILD_TESTS=ON

cmake --build build\windows-x64-cuda\openEMS --parallel
ctest --test-dir build\windows-x64-cuda\openEMS -C Release --output-on-failure
cmake --install build\windows-x64-cuda\openEMS
```

For a distributable binary that covers several NVIDIA GPU generations, pass a
semicolon-separated architecture list instead of `89`. This increases compile
time and binary size.

## Select the CUDA engine

The standalone executable uses the CPU engine unless CUDA is requested:

```powershell
openEMS.exe model.xml --engine=cuda --cuda-device=0
```

`--cuda-debug-sync` synchronizes after every timestep and is useful for finding
kernel failures. It should not be enabled for performance measurements.

Python forwards keyword arguments to the same option parser, replacing
underscores with dashes:

```python
fdtd.Run(
    sim_path,
    engine="cuda",
    cuda_device=0,
    cuda_debug_sync=False,
)
```

For Python, set `OPENEMS_INSTALL_PATH` to the CUDA build's `bin` directory and
`CSXCAD_INSTALL_PATH` to the dependency installation's `bin` directory before
importing `openEMS`. The standalone executable needs both directories plus the
CUDA runtime directory on `PATH`.

## Supported model features

The current CUDA engine supports:

- Cartesian Yee grids, including nonuniform and anisotropic X/Y/Z spacing
- Single-precision electric and magnetic field updates resident on one GPU
- Isotropic material permittivity, permeability, and conductivity represented
  by the base openEMS update coefficients
- PEC and PMC boundaries
- Voltage and current excitation, including per-cell delay and periodic signals
- TFSF plane-wave excitation, including fractional delays and numerical phase
  velocity compensation
- UPML boundaries
- Parallel resistive/RC loads and dynamic series/parallel lumped RLC elements
- Periodic-excitation steady-state detection using device-side probe records
  and energy reduction
- Existing probes, field dumps, NF2FF, SAR, and other processing that reads the
  normal CPU engine interface

Processing compatibility is implemented with a pinned host mirror. Device
fields are copied to the host only at processing timesteps selected by openEMS.
Output-heavy simulations can therefore become PCIe- or CPU-limited even when
the FDTD kernels are much faster.

Frequency-domain dump types 10 through 15 accumulate fields during the run and
write only the final complex fields. The following optional custom dump
attributes support periodic dataset workflows while preserving legacy dump
behavior by default:

- `FDInterval`: timestep interval between samples.
- `FDFirstSample` and `FDLastSample`: one-based inclusive sample window.
- `FDNormalization=periodic-average`: divide the complex sum by its sample
  count instead of applying the legacy single-sided integral scaling.

Frequency-domain phase and accumulation use double precision; the underlying
FDTD fields remain single precision. Processing is not scheduled before the
first requested sample, avoiding unnecessary accelerator synchronization and
host field copies during the skipped portion of the simulation.

The base device allocation is 72 bytes per Yee cell: six field components plus
twelve update-coefficient components, all stored as `float`. UPML and dynamic
lumped elements allocate additional device state. CPU-side operator and mirror
storage still exists.

## Deliberately rejected features

The engine stops during setup with a named error for active extensions that
have not been ported. Use `--engine=multithreaded` for models containing:

- Cylindrical coordinates or cylindrical multigrid
- Mur/local absorbing boundary conditions
- Lorentz, Drude, or Debye dispersive material updates
- Conducting-sheet or absorbing-sheet extensions

Rejecting these cases is intentional: silently running only the base Yee update
would produce plausible-looking but physically incorrect results.

## Validation and current benchmark

The regression generator is
`FDTD/tests/generate_cuda_smoke.py`. The repeatable regression runner is:

```powershell
python openEMS\FDTD\tests\run_cuda_regression.py `
  build\windows-x64-cuda\openEMS\openEMS.exe `
  build\cuda-regression --cuda-debug-sync
```

CPU/CUDA comparisons cover deliberately anisotropic meshes, excitation,
six-sided UPML, a parallel resistor, a series RLC load, and a sinusoidal TFSF
plane wave with steady-state detection. Probe timestamps match exactly; the
largest observed probe error in these tests was approximately `1.34e-5`
relative to the CPU peak.

On the development system (GeForce RTX 4090, CUDA 13.3), a 129 x 129 x 129
anisotropic Cartesian mesh run for 512 timesteps achieved the following after
fixing the CPU worker count at its measured optimum of 16:

| Engine | Time stepping | Throughput |
| --- | ---: | ---: |
| CUDA | 0.36 s | 3013 MCells/s |
| Multithreaded compressed SSE, 16 workers | 1.45 s | 759 MCells/s |

This is approximately a 4x time-stepping speedup for that homogeneous
benchmark. It includes normal probe processing but excludes geometry/operator
creation. Automatic CPU worker selection starts at one worker and can
understate CPU performance in simulations that finish before its first tuning
interval. Performance also varies with model size, extensions, output cadence,
CPU, and GPU.

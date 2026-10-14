# Metal backend (development preview)

Enable the optional macOS Apple Silicon compute backend with
`-DOPENEMS_ENABLE_METAL=ON`. Select it at runtime with `--engine=metal`.
The CUDA and CPU implementations remain unchanged.

## Design

The Metal backend runs the Cartesian E/H Yee-grid updates using two Metal
compute kernels. Existing CPU-side openEMS extensions are invoked at the
voltage/current update boundaries, including UPML, excitation, TFSF, lumped
elements, Mur, and dispersive materials. This is a **hybrid correctness-first
backend**, not a full GPU port of each CUDA extension; it synchronizes field
buffers twice per timestep and may be slower than CPU on small workloads.
Cylindrical models are rejected.

GPU fields are single precision. RLC coefficients and history remain double
precision in the shared CPU extension; they are not stored in Metal buffers.
The extension dispatcher explicitly routes Metal to the host `Engine` accessors.
Without that route, dispatched extensions throw `unknown engine` before they
can update the fields. Metal field-update coefficients are uploaded once at
initialization. Runtime shader compilation requires a working Metal device.

## Fast smoke test (no openEMS installation required)

On macOS with Xcode Command Line Tools and Boost headers installed:

```sh
cmake -S scripts/verification -B build/metal-compile-smoke
cmake --build build/metal-compile-smoke --parallel
cmake --build build/metal-compile-smoke --target test
```

The CMake smoke target compiles the Metal backend and both shared RLC
translation units. Two tests check the production extension dispatcher (using
small engine stand-ins) and compile the embedded Metal kernels/create both
compute pipelines. On 2026-10-14, AppleClang 17 compilation and both tests passed
(2/2). The dispatcher test fails with `unknown engine` against the original
header. These checks **do not** run a physical FDTD simulation or verify its
numerical correctness.

## Numerical comparison (requires a complete openEMS build)

```sh
python3 scripts/verification/verify_metal.py \
  --binary /absolute/path/to/openEMS --work build/verification
```

It generates deterministic Cartesian models with the existing generator,
runs the CPU `basic` and `metal` engines, and compares probe timestamps
and waveforms using a default relative tolerance of 1e-4.

Coverage includes base grid, UPML, resistor, series RLC and TFSF. Additional
CPU extensions should be validated on dedicated physical models before
production use. No benchmark or end-to-end accuracy claims are made before
the full build and test suite completes.

The longer RLC precision fixture needs only a complete Metal-enabled solver and
Python 3.8+ (no Python bindings). From current `main`, run it in a new directory:

```sh
python3 openEMS/FDTD/tests/run_rlc_precision.py build/rlc-metal \
  --binary /absolute/path/to/openEMS --engine basic --engine metal --timeout 1800
```

This checks 786,432 steps against CPU with the same stability guards and error
thresholds as the CUDA reproducer. The timeout is per process and can be raised
for the hybrid backend. **This CPU/Metal numerical comparison has not yet been
run**; the smoke build does not provide a complete solver executable. See
[`RLC_PRECISION.md`](../../openEMS/FDTD/tests/RLC_PRECISION.md).

## Dependencies

Install the ordinary openEMS build dependencies (fparser, CSXCAD, TinyXML,
HDF5, VTK, Boost), plus AppleClang/Metal frameworks, before configuring
the solver. The upstream Python model generator also needs openEMS Python
bindings, CSXCAD Python bindings and NumPy.

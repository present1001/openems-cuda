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

GPU fields are single precision; CPU extensions retain their existing
precision and semantics. Metal coefficients are uploaded once at engine
initialization. Runtime shader compilation requires a working Metal device.

## Fast smoke test (no openEMS installation required)

On macOS with Xcode Command Line Tools and Boost headers installed:

```sh
cmake -S scripts/verification -B build/metal-compile-smoke
cmake --build build/metal-compile-smoke --parallel
cmake --build build/metal-compile-smoke --target test
```

The CMake smoke target compiles the Metal backend's C++ and Objective-C++
translation units. The runtime test compiles the two embedded Metal kernels
from `engine_metal.mm` and creates both compute pipelines. This **does not**
run a physical FDTD simulation or verify its numerical correctness.

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

## Dependencies

Install the ordinary openEMS build dependencies (fparser, CSXCAD, TinyXML,
HDF5, VTK, Boost), plus AppleClang/Metal frameworks, before configuring
the solver. The upstream Python model generator also needs openEMS Python
bindings, CSXCAD Python bindings and NumPy.

#!/usr/bin/env python3
"""Compare deterministic CPU and Metal physics probes."""
import argparse
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "openEMS/FDTD/tests/generate_cuda_smoke.py"
CASES = {
    "base": [],
    "upml": ["--pml"],
    "resistor": ["--pml", "--lumped", "resistor"],
    "series_rlc": ["--pml", "--lumped", "series-rlc"],
    "tfsf": ["--pml", "--tfsf", "--sinusoidal"],
}
def run(cmd, cwd):
    subprocess.run([str(x) for x in cmd], cwd=cwd, check=True)
def probe(path):
    return [tuple(float(v) for v in line.split()[:2])
            for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("%")]
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--work", type=Path, default=ROOT / "build/verification")
    p.add_argument("--tolerance", type=float, default=1e-4)
    args = p.parse_args()
    binary = args.binary.resolve()
    for name, flags in CASES.items():
        folder = args.work.resolve() / name
        folder.mkdir(parents=True, exist_ok=True)
        model = folder / "model.xml"
        run([sys.executable, GENERATOR, model, *flags], folder)
        values = []
        for engine in ("basic", "metal"):
            dst = folder / engine
            dst.mkdir(exist_ok=True)
            run([binary, model, "--engine=" + engine, "--verbose=0", "--disable-dumps"], dst)
            values.append(probe(dst / "cuda_smoke_probe"))
        cpu, gpu = values
        if not cpu or len(cpu) != len(gpu):
            raise RuntimeError(name + ": missing or unequal output")
        if any(a[0] != b[0] for a,b in zip(cpu,gpu)):
            raise RuntimeError(name + ": inconsistent sample times")
        peak = max(abs(v) for _,v in cpu)
        error = max(abs(v-w) for (_,v),(_,w) in zip(cpu,gpu)) / max(peak, sys.float_info.min)
        print(name, "relative error", error, flush=True)
        if not math.isfinite(error) or error > args.tolerance:
            raise RuntimeError(name + ": comparison failed")
if __name__ == "__main__":
    main()

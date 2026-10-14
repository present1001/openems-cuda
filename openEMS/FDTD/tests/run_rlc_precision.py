"""Check the small PEC RLC precision reproducer (Python 3.8+, standard library)."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET


FIXTURE = Path(__file__).with_name("fixtures") / "rlc_precision.xml"
PROBES = ("load_voltage", "field_voltage")


def read_probe(path):
    samples = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("%"):
            continue
        time, value = map(float, line.split()[:2])
        if not math.isfinite(time) or not math.isfinite(value):
            raise ValueError("Nonfinite probe sample: " + str(path))
        if samples and time <= samples[-1][0]:
            raise ValueError("Probe timestamps do not increase: " + str(path))
        samples.append((time, value))
    if len(samples) < 2:
        raise ValueError("Missing probe samples: " + str(path))
    return samples


def check_run(directory, engine, steps, dt):
    log = (directory / "solver.log").read_text(errors="replace")
    completed = re.findall(r"Time for\s+(\d+)\s+iterations", log)
    if not completed or int(completed[-1]) != steps:
        raise ValueError("The solver did not complete the requested timesteps")
    if engine == "cuda" and "Create CUDA FDTD engine on device" not in log:
        raise ValueError("CUDA execution was not confirmed")
    if re.search(r"Energy:\s*~?\s*[+-]?(?:nan|inf)\b", log, re.IGNORECASE):
        raise ValueError("Nonfinite solver energy")
    metrics = {}
    for name in PROBES:
        samples = read_probe(directory / name)
        # The fixture samples about every 0.68 ps; allow one sample at the end.
        if abs(samples[0][0]) > dt or abs(samples[-1][0] - steps * dt) > 1e-12:
            raise ValueError("Incomplete probe observation: " + name)
        peak = max(abs(value) for _, value in samples)
        tail = max(abs(value) for _, value in samples[-max(1, len(samples) // 10):])
        # Fixture-specific guard: reject both an unexcited and an unstable run.
        if not 1e-6 < peak < 1e-3:
            raise ValueError("Unexpected peak voltage for {}: {}".format(name, peak))
        if tail / peak >= 0.25:
            raise ValueError("Late-time voltage did not decay: " + name)
        metrics[name] = dict(samples=len(samples), peak_v=peak,
                             tail_over_peak=tail / peak)
    return metrics


def compare(cpu, cuda):
    if len(cpu) != len(cuda):
        raise ValueError("CPU/CUDA sample counts differ")
    if any(abs(a[0] - b[0]) > 1e-18 for a, b in zip(cpu, cuda)):
        raise ValueError("CPU/CUDA timestamps differ")
    peak = max(abs(value) for _, value in cpu)
    if peak == 0:
        raise ValueError("Unexcited CPU reference")
    errors = [abs(a[1] - b[1]) / peak for a, b in zip(cpu, cuda)]
    maximum = max(errors)
    rms = math.sqrt(sum(error * error for error in errors) / len(errors))
    if maximum > 2e-3 or rms > 5e-4:
        raise ValueError("CPU/CUDA waveform mismatch: max={}, rms={}".format(maximum, rms))
    return dict(max_normalized_error=maximum, rms_normalized_error=rms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_directory", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--engine", choices=("basic", "cuda"), action="append")
    parser.add_argument("--check-existing", action="store_true",
                        help="Check archived solver.log and probe files without running a solver")
    args = parser.parse_args()
    if not args.check_existing and (not args.binary or not args.binary.is_file()):
        parser.error("Provide --binary pointing to a built openEMS executable")
    engines = list(dict.fromkeys(args.engine or ("basic", "cuda")))
    work = args.work_directory.resolve()
    if args.check_existing and not work.is_dir():
        parser.error("The existing work directory does not exist")
    model = ET.parse(FIXTURE).getroot().find("FDTD")
    steps, dt = int(model.get("NumberOfTimesteps")), float(model.get("TimeStep"))
    if not args.check_existing:
        work.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(FIXTURE, work / "model.xml")
    report = dict(fixture_sha256=hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
                  timesteps=steps, timestep_s=dt, runs={}, comparisons={})
    failed = False
    for engine in engines:
        directory = work / engine
        try:
            if not args.check_existing:
                directory.mkdir()
                command = [str(args.binary.resolve()), str(work / "model.xml"),
                           "--engine=" + engine, "--numThreads=1", "--disable-dumps",
                           "--verbose=1"]
                # CUDA_VISIBLE_DEVICES is inherited; use its first visible device.
                if engine == "cuda":
                    command.append("--cuda-device=0")
                with (directory / "solver.log").open("w") as log:
                    subprocess.run(command, cwd=directory, stdout=log,
                                   stderr=subprocess.STDOUT, check=True, timeout=300)
            metrics = check_run(directory, engine, steps, dt)
            report["runs"][engine] = dict(passed=True, probes=metrics)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            report["runs"][engine] = dict(passed=False, error=str(error))
            failed = True
    if not failed and set(engines) == {"basic", "cuda"}:
        for name in PROBES:
            try:
                report["comparisons"][name] = compare(
                    read_probe(work / "basic" / name), read_probe(work / "cuda" / name))
            except (OSError, ValueError) as error:
                report["comparisons"][name] = dict(error=str(error))
                failed = True
    report["passed"] = not failed
    (work / "rlc_precision_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run deterministic CPU/CUDA probe comparisons for supported physics paths."""

import argparse
import math
import subprocess
import sys
from pathlib import Path


CASES = (
    ("base", ()),
    ("upml", ("--pml",)),
    ("upml_resistor", ("--pml", "--lumped", "resistor")),
    ("upml_series_rlc", ("--pml", "--lumped", "series-rlc")),
    ("tfsf_sinus_upml", ("--pml", "--tfsf", "--sinusoidal")),
)


def run(command, cwd):
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode:
        print(result.stdout, file=sys.stderr)
        raise RuntimeError(
            "Command failed with exit code {}: {}".format(
                result.returncode, " ".join(map(str, command))
            )
        )
    return result.stdout


def read_probe(path):
    samples = []
    with path.open("r", encoding="utf-8") as probe:
        for line in probe:
            if not line.strip() or line.startswith("%"):
                continue
            time, value = line.split()[:2]
            samples.append((float(time), float(value)))
    return samples


def compare(cpu, cuda):
    if len(cpu) != len(cuda):
        raise RuntimeError(
            "Probe sample counts differ: CPU={} CUDA={}".format(
                len(cpu), len(cuda)
            )
        )

    max_time_difference = 0.0
    max_absolute_error = 0.0
    squared_error = 0.0
    cpu_peak = 0.0
    for cpu_sample, cuda_sample in zip(cpu, cuda):
        max_time_difference = max(
            max_time_difference, abs(cpu_sample[0] - cuda_sample[0])
        )
        difference = abs(cpu_sample[1] - cuda_sample[1])
        max_absolute_error = max(max_absolute_error, difference)
        squared_error += difference * difference
        cpu_peak = max(cpu_peak, abs(cpu_sample[1]))

    scale = max(cpu_peak, sys.float_info.min)
    return {
        "samples": len(cpu),
        "max_time_difference": max_time_difference,
        "max_relative_error": max_absolute_error / scale,
        "rms_relative_error": math.sqrt(squared_error / max(len(cpu), 1))
        / scale,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("openems_executable", type=Path)
    parser.add_argument("work_directory", type=Path)
    parser.add_argument("--timesteps", type=int, default=256)
    parser.add_argument("--relative-tolerance", type=float, default=1e-4)
    parser.add_argument("--cuda-debug-sync", action="store_true")
    args = parser.parse_args()

    executable = args.openems_executable.resolve()
    if not executable.is_file():
        parser.error("openEMS executable does not exist: {}".format(executable))

    generator = Path(__file__).with_name("generate_cuda_smoke.py").resolve()
    work_directory = args.work_directory.resolve()
    work_directory.mkdir(parents=True, exist_ok=True)

    failed = False
    for name, model_options in CASES:
        case_directory = work_directory / name
        case_directory.mkdir(parents=True, exist_ok=True)
        model = case_directory / "model.xml"
        run(
            [
                sys.executable,
                str(generator),
                str(model),
                "--timesteps",
                str(args.timesteps),
                *model_options,
            ],
            case_directory,
        )

        for engine in ("basic", "cuda"):
            run_directory = case_directory / engine
            run_directory.mkdir(parents=True, exist_ok=True)
            command = [
                str(executable),
                str(model),
                "--engine={}".format(engine),
                "--verbose=0",
                "--disable-dumps",
            ]
            if engine == "cuda" and args.cuda_debug_sync:
                command.append("--cuda-debug-sync")
            output = run(command, run_directory)
            (run_directory / "run.log").write_text(output, encoding="utf-8")

        metrics = compare(
            read_probe(case_directory / "basic" / "cuda_smoke_probe"),
            read_probe(case_directory / "cuda" / "cuda_smoke_probe"),
        )
        passed = (
            metrics["max_time_difference"] == 0.0
            and metrics["max_relative_error"] <= args.relative_tolerance
        )
        failed = failed or not passed
        print(
            "{:<18} {} samples={} max_rel={:.6g} rms_rel={:.6g}".format(
                name,
                "PASS" if passed else "FAIL",
                metrics["samples"],
                metrics["max_relative_error"],
                metrics["rms_relative_error"],
            )
        )

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

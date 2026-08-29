"""Generate a small Cartesian model for CPU/CUDA regression testing."""

import argparse
from pathlib import Path

import numpy as np

from CSXCAD import ContinuousStructure
from openEMS import openEMS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    boundary = parser.add_mutually_exclusive_group()
    boundary.add_argument("--pml", action="store_true")
    boundary.add_argument("--mur", action="store_true")
    parser.add_argument(
        "--lumped", choices=("none", "resistor", "series-rlc"), default="none"
    )
    parser.add_argument("--tfsf", action="store_true")
    parser.add_argument("--sinusoidal", action="store_true")
    parser.add_argument("--cells", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=256)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fdtd = openEMS(NrTS=args.timesteps, EndCriteria=0)
    if args.sinusoidal:
        fdtd.SetSinusExcite(5e9)
    else:
        fdtd.SetGaussExcite(0, 3e9)
    boundary_condition = "PML_2" if args.pml else "MUR" if args.mur else "PEC"
    fdtd.SetBoundaryCond([boundary_condition] * 6)

    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    grid = csx.GetGrid()
    grid.SetDeltaUnit(1e-3)

    # Deliberately anisotropic: dx=2 mm, dy=3 mm, dz=1 mm.
    if args.cells:
        logical = np.arange(args.cells + 1, dtype=float) - args.cells // 2
        grid.SetLines("x", 2 * logical)
        grid.SetLines("y", 3 * logical)
        grid.SetLines("z", logical)
    else:
        grid.SetLines("x", np.arange(-10, 12, 2, dtype=float))
        grid.SetLines("y", np.arange(-12, 15, 3, dtype=float))
        grid.SetLines("z", np.arange(-6, 7, 1, dtype=float))

    if args.tfsf:
        excitation = csx.AddExcitation(
            "plane_wave", exc_type=10, exc_val=[1, 0, 0]
        )
        excitation.SetPropagationDir([0, 0, -1])
        excitation.SetFrequency(5e9)
        excitation.AddBox(start=[-6, -6, -4], stop=[6, 6, 4])
    else:
        excitation = csx.AddExcitation(
            "z_soft_excitation", exc_type=0, exc_val=[0, 0, 1]
        )
        excitation.AddBox(start=[0, 0, -1], stop=[0, 0, 1])

    if args.lumped == "resistor":
        load = csx.AddLumpedElement(
            "parallel_resistor", ny="z", caps=False, R=50.0, LEtype=0
        )
        load.AddBox(start=[4, 0, -1], stop=[4, 0, 1], priority=20)
    elif args.lumped == "series-rlc":
        load = csx.AddLumpedElement(
            "series_rlc",
            ny="z",
            caps=False,
            R=10.0,
            L=1e-9,
            C=2.5e-12,
            LEtype=1,
        )
        load.AddBox(start=[4, 0, -1], stop=[4, 0, 1], priority=20)

    probe = csx.AddProbe("cuda_smoke_probe", p_type=0)
    if args.tfsf:
        probe.AddBox(start=[0, 0, 0], stop=[2, 0, 0])
    else:
        probe.AddBox(start=[4, 0, -1], stop=[4, 0, 1])

    fdtd.Write2XML(str(args.output.resolve()))


if __name__ == "__main__":
    main()

"""Run the original nine-door MuJoCo arena in paired trajectory modes."""
import argparse
import json
import os
from pathlib import Path

from comparison_log import ComparisonLog
from run_demo import run

SCENARIOS = {"auto": "auto", "detour": "detour"}


def record(output, scenario, solver, seed):
    folder = output / scenario / solver
    folder.mkdir(parents=True, exist_ok=True)
    log = ComparisonLog()
    with (folder / "trace.jsonl").open("w", encoding="utf-8") as trace:
        result = run(mode=SCENARIOS[scenario], trajectory_opt_mode=solver, seed=seed,
                     trace=trace, summary_file=folder / "result.json", comparison=log)
    for name, rows in (("plans", log.plans), ("frames", log.frames), ("astar", log.astar),
                       ("qp_attempts", log.qp_attempts)):
        (folder / (name+".json")).write_text(json.dumps(rows), encoding="utf-8")
    print(scenario, solver, result["reason"], result["seconds"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/comparison"))
    parser.add_argument("--scenario", choices=SCENARIOS, action="append")
    parser.add_argument("--solver", choices=("baseline", "joint"), action="append")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    scenarios = args.scenario or tuple(SCENARIOS)
    if not args.plot_only:
        for scenario in scenarios:
            for solver in args.solver or ("baseline", "joint"):
                record(args.output, scenario, solver, args.seed)
    if not args.no_plots:
        cache = args.output / "matplotlib_cache"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache.resolve()))
        from comparison_plots import plot_comparison
        plot_comparison(args.output, scenarios)


if __name__ == "__main__":
    main()

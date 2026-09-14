"""
Experiment runner for the JSSP genetic algorithm.

Runs each benchmark × 3 parameter sets × N_RUNS times (GA is random),
then writes metrics.csv, a time table, Gantt charts, and one comparison bar chart.

Depends on jobshop.py.
"""

import csv
import random
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt

from jobshop import GeneticAlgorithm, load_data

# Assignment: 3 size categories × 2 Lawrence instances each
INSTANCES = {
    "small":  ["data/la01.txt", "data/la05.txt"],   # 10×5
    "medium": ["data/la17.txt", "data/la20.txt"],   # 10×10
    "large":  ["data/la32.txt", "data/la35.txt"],   # 30×10
}

# Three GA configurations to compare in the report
PARAM_SETS = {
    "A": dict(population_size=50,  generations=200,  crossover_rate=0.8, mutation_rate=0.1),
    "B": dict(population_size=100, generations=500,  crossover_rate=0.8, mutation_rate=0.2),
    "C": dict(population_size=150, generations=1000, crossover_rate=0.9, mutation_rate=0.3),
}

N_RUNS = 10  # repeats, assignment asks 10–30


def convergence_generation(history):
    """First generation to include best makespan."""
    if not history:
        return 0
    best = min(history)
    for gen, value in enumerate(history, start=1):
        if value == best:
            return gen
    return len(history)


def one_run(instance_path, params, seed):
    """One full GA run → makespan, chromosome, time, convergence gen."""
    random.seed(seed)
    ga = GeneticAlgorithm(load_data(instance_path), **params)

    t0 = time.perf_counter()
    best_entry, history = ga.run(verbose=False)

    # best_entry is [chromosome, makespan]
    return {
        "makespan": best_entry[1],
        "chromosome": best_entry[0],
        "time": time.perf_counter() - t0,
        "convergence_gen": convergence_generation(history),
    }


def summarize(runs):
    """Assignment metrics across the N_RUNS repeats."""
    ms = [r["makespan"] for r in runs]
    return {
        "best": min(ms),
        "worst": max(ms),
        "avg": statistics.mean(ms),
        "std": statistics.stdev(ms) if len(ms) > 1 else 0.0,
        "avg_time": statistics.mean(r["time"] for r in runs),
        "avg_convergence": statistics.mean(r["convergence_gen"] for r in runs),
    }


def plot_gantt(schedule, makespan, title, out_path):
    """One row per machine; bars = operations (semi-active schedule)."""
    machines = sorted({op["machine"] for op in schedule})
    fig, ax = plt.subplots(figsize=(12, 4 + 0.4 * len(machines)))
    colors = plt.cm.tab20.colors

    for op in schedule:
        y = machines.index(op["machine"])
        w = op["finish"] - op["start"]
        ax.barh(y, w, left=op["start"], height=0.6,
                color=colors[op["job"] % len(colors)], edgecolor="black", linewidth=0.3)
        ax.text(op["start"] + w / 2, y, f"J{op['job']}", ha="center", va="center", fontsize=7)

    ax.set_yticks(range(len(machines)), [f"M{m}" for m in machines])
    ax.set_xlabel("Time")
    ax.set_title(f"{title}  (Cmax={makespan})")
    ax.set_xlim(0, makespan)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_reports(rows, out_dir="results"):
    """Time pivot (instance × A/B/C) + one bar chart (best makespan & avg time)."""
    instances = list(dict.fromkeys(r["instance"] for r in rows))
    lookup = {(r["instance"], r["param_set"]): r for r in rows}

    # Assignment time table
    time_rows = [
        {
            "instance": inst,
            "time_A_s": round(float(lookup[(inst, "A")]["avg_time"]), 4),
            "time_B_s": round(float(lookup[(inst, "B")]["avg_time"]), 4),
            "time_C_s": round(float(lookup[(inst, "C")]["avg_time"]), 4),
        }
        for inst in instances
    ]
    with open(f"{out_dir}/time_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=time_rows[0].keys())
        w.writeheader()
        w.writerows(time_rows)

    # Side-by-side bars for report figures
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x, width = range(len(instances)), 0.25
    for ax, field, title, fmt in [
        (axes[0], "best", "Best makespan", "%.0f"),
        (axes[1], "avg_time", "Avg time (s)", "%.2f"),
    ]:
        for i, s in enumerate("ABC"):
            vals = [float(lookup[(inst, s)][field]) for inst in instances]
            bars = ax.bar([xi + (i - 1) * width for xi in x], vals, width, label=f"Set {s}")
            ax.bar_label(bars, fmt=fmt, fontsize=7, padding=2)
        ax.set_xticks(list(x), instances)
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/bar_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {out_dir}/time_table.csv and {out_dir}/bar_comparison.png")


def run_all(out_csv="results/metrics.csv", gantt_dir="results/gantt"):
    """Sweep all instances × param sets; write CSV, Gantts, and reports."""
    Path("results").mkdir(exist_ok=True)
    Path(gantt_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    for category, paths in INSTANCES.items():
        for path in paths:
            name = Path(path).stem  # e.g. "la01"
            for set_name, params in PARAM_SETS.items():
                runs = [one_run(path, params, seed=1000 + i) for i in range(N_RUNS)]
                stats = summarize(runs)
                print(name, set_name, stats)

                rows.append({
                    "category": category,
                    "instance": name,
                    "param_set": set_name,
                    **stats,
                    **{f"p_{k}": v for k, v in params.items()},
                })

                # Gantt for the best of the N_RUNS repeats
                best = min(runs, key=lambda r: r["makespan"])
                schedule, cmax = GeneticAlgorithm(load_data(path)).build_schedule(best["chromosome"])
                plot_gantt(schedule, cmax, f"{name} / set {set_name}",
                           f"{gantt_dir}/{name}_{set_name}.png")

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv}")  # has best/worst/avg/std/time/convergence
    save_reports(rows)


if __name__ == "__main__":
    run_all()

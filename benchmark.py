"""
Experiment runner for the JSSP genetic algorithm.

One-factor parameter sweep: baseline A, then B/C/D each change one knob.
6 instances x 4 param sets x N_RUNS -> metrics, time table, Gantt, convergence.

Depends on jobshop.py.
"""

import csv
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt

from jobshop import GeneticAlgorithm, load_data

# -----AI Generated start-----
# Run on half of the available CPU cores
NUM_WORKERS = os.cpu_count() // 2 or 1
# -----AI Generated end-----

# 3 size categories x 2 Lawrence instances each
INSTANCES = {
    "small":  ["data/la01.txt", "data/la05.txt"],   # 10x5
    "medium": ["data/la17.txt", "data/la20.txt"],   # 10x10
    "large":  ["data/la32.txt", "data/la35.txt"],   # 30x10
}

# One-factor-at-a-time configs (gens fixed; mut kept in 0.03-0.10)
# A = baseline; B = mutation only; C = crossover only; D = population only
PARAM_SETS = {
    "A": dict(population_size=100, generations=1250, crossover_rate=0.8, mutation_rate=0.05),
    "B": dict(population_size=100, generations=1250, crossover_rate=0.8, mutation_rate=0.10),
    "C": dict(population_size=100, generations=1250, crossover_rate=0.6, mutation_rate=0.05),
    "D": dict(population_size=150, generations=1250, crossover_rate=0.8, mutation_rate=0.05),
}

N_RUNS = 30


def convergence_generation(history):
    """First generation to include best makespan."""
    if not history:
        return 0
    best = min(history)
    for gen, value in enumerate(history, start=1):
        if value == best:
            return gen
    return len(history)


def one_run(instance_path, params):
    """One full GA run → makespan, chromosome, time, convergence gen, history."""
    ga = GeneticAlgorithm(load_data(instance_path), **params)

    t0 = time.perf_counter()
    best_entry, history = ga.run(verbose=False)

    # best_entry is [chromosome, makespan]
    # history[i] = best makespan found up to generation i+1
    return {
        "makespan": best_entry[1],
        "chromosome": best_entry[0],
        "time": time.perf_counter() - t0,
        "convergence_gen": convergence_generation(history),
        "history": history,
    }

# -----AI Generated start-----
def _worker_task(args):
    """Worker function executed inside parallel process pool."""
    category, path, set_name, params, run_idx = args
    run_result = one_run(path, params)
    return {
        "category": category,
        "instance": Path(path).stem,
        "path": path,
        "param_set": set_name,
        "run_idx": run_idx,
        **run_result,
    }
# -----AI Generated end-----

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
    cmap = plt.cm.tab20
    colors = [cmap(i) for i in range(cmap.N)]

    for op in schedule:
        y = machines.index(op["machine"])
        w = op["finish"] - op["start"]
        ax.barh(y, w, left=op["start"], height=0.6,
                color=colors[op["job"] % len(colors)], edgecolor="black", linewidth=0.3)
        if makespan > 0 and (w / makespan) >= 0.02:
            ax.text(op["start"] + w / 2, y, f"J{op['job']}", ha="center", va="center", fontsize=7)

    ax.set_yticks(range(len(machines)), [f"M{m}" for m in machines])
    ax.set_xlabel("Time")
    ax.set_title(f"{title}  (Cmax={makespan})")
    ax.set_xlim(0, makespan)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_convergence(histories, title, out_path):
    """Line chart: best makespan so far vs generation (one line per param set)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for set_name, history in histories.items():
        ax.plot(range(1, len(history) + 1), history, label=f"Set {set_name}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best makespan so far")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_reports(rows, out_dir="results"):
    """Time pivot (instance x each param set) + bar chart (best makespan & avg time)."""
    instances = list(dict.fromkeys(r["instance"] for r in rows))
    set_names = list(PARAM_SETS.keys())
    lookup = {(r["instance"], r["param_set"]): r for r in rows}

    time_rows = []
    for inst in instances:
        row = {"instance": inst}
        for s in set_names:
            row[f"time_{s}_s"] = round(float(lookup[(inst, s)]["avg_time"]), 4)
        time_rows.append(row)

    with open(f"{out_dir}/time_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=time_rows[0].keys())
        w.writeheader()
        w.writerows(time_rows)

    n_sets = len(set_names)
    width = 0.8 / n_sets
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    x = range(len(instances))
    for ax, field, title, fmt in [
        (axes[0], "best", "Best makespan", "%.0f"),
        (axes[1], "avg_time", "Avg time (s)", "%.2f"),
    ]:
        for i, s in enumerate(set_names):
            vals = [float(lookup[(inst, s)][field]) for inst in instances]
            offsets = [xi + (i - (n_sets - 1) / 2) * width for xi in x]
            bars = ax.bar(offsets, vals, width, label=f"Set {s}")
            ax.bar_label(bars, fmt=fmt, fontsize=6, padding=2)
        ax.set_xticks(list(x), instances)
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/bar_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {out_dir}/time_table.csv and {out_dir}/bar_comparison.png")


def run_all(out_csv="results/metrics.csv", gantt_dir="results/gantt", num_workers=NUM_WORKERS):
    """Sweep all instances x param sets in parallel; write CSV, Gantts, and reports."""
    Path("results").mkdir(exist_ok=True)
    Path(gantt_dir).mkdir(parents=True, exist_ok=True)
    conv_dir = Path("results/convergence")
    conv_dir.mkdir(exist_ok=True)

    # -----AI Generated start-----
    tasks = []
    for category, paths in INSTANCES.items():
        for path in paths:
            for set_name, params in PARAM_SETS.items():
                for i in range(N_RUNS):
                    tasks.append((category, path, set_name, params, i))

    total_tasks = len(tasks)
    print(f"Starting {total_tasks} runs using {num_workers} parallel workers...")
    t_start = time.perf_counter()

    results_by_config = {}
    completed_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_worker_task, t) for t in tasks]
        for future in as_completed(futures):
            res = future.result()
            key = (res["category"], res["instance"], res["param_set"])
            results_by_config.setdefault(key, []).append(res)
            completed_count += 1
            print(
                f"[{completed_count:3d}/{total_tasks}] "
                f"{res['instance']} Set {res['param_set']} "
                f"(Run {res['run_idx']+1:2d}/{N_RUNS}) -> "
                f"Makespan: {res['makespan']}, Time: {res['time']:.2f}s"
            )

    elapsed_all = time.perf_counter() - t_start
    print(f"All {total_tasks} runs finished in {elapsed_all:.2f}s across {num_workers} cores.")
    # -----AI Generated end-----

    rows = []
    for category, paths in INSTANCES.items():
        for path in paths:
            name = Path(path).stem  # e.g. "la01"
            best_histories = {}  # param set → history from best of N_RUNS
            for set_name, params in PARAM_SETS.items():
                key = (category, name, set_name)
                runs = results_by_config[key]
                runs.sort(key=lambda r: r["run_idx"])  # keep deterministic ordering

                stats = summarize(runs)
                print(name, set_name, stats)

                rows.append({
                    "category": category,
                    "instance": name,
                    "param_set": set_name,
                    **stats,
                    **{f"p_{k}": v for k, v in params.items()},
                })

                # Gantt + keep convergence curve from the best repeat
                best = min(runs, key=lambda r: r["makespan"])
                best_histories[set_name] = best["history"]
                schedule, cmax = GeneticAlgorithm(load_data(path)).build_schedule(best["chromosome"])
                plot_gantt(schedule, cmax, f"{name} / set {set_name}",
                           f"{gantt_dir}/{name}_{set_name}.png")

            plot_convergence(
                best_histories,
                f"{name}: best makespan over generations",
                conv_dir / f"{name}.png",
            )

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv}")  # has best/worst/avg/std/time/convergence
    save_reports(rows)


if __name__ == "__main__":
    run_all()

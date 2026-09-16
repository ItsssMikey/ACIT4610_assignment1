# Job Shop Scheduling (JSSP) — Genetic Algorithm

## Setup
For pip:
```bash
pip install -r requirements.txt
```

## Run

Run the full benchmark suite across instances and parameter sets:

```bash
uv run benchmark.py
```
OR
```bash
python benchmark.py
```

### Parameters (`benchmark.py`)
Configured at the top of `benchmark.py`:
- `NUM_WORKERS`: Number of parallel CPU processes (default: `half of system cores`)
- `N_RUNS`: Number of repeated runs per instance (default: `10`)
- `INSTANCES`: Benchmark files to test from `data/`
- `PARAM_SETS`: Parameter dictionaries to evaluate (A, B, C)

### Outputs (`results/`)
- `metrics.csv`: Summary metrics (best/avg/worst/std makespan, runtime, convergence gen)
- `time_table.csv`: Runtime comparison across parameter sets
- `bar_comparison.png`: Side-by-side comparison charts for makespan and runtime
- `gantt/*.png`: Schedule Gantt charts of the best runs

---

## Single Run (Testing Only)

Run a single instance directly via `jobshop.py`:

```bash
uv run jobshop.py
# or: python jobshop.py
```

### Parameters (`jobshop.py`)
Configured in the execution block at the bottom of `jobshop.py`:
```python
instance = load_data("./data/la35.txt")  # Benchmark file in data/
jssp_solver = GeneticAlgorithm(
    instance,
    population_size = 100,
    generations     = 1250,
    mutation_rate   = 0.05,
    crossover_rate  = 0.8,
    tournament_size = 2,
    seed = 42
)
```

---

## AI Usage Declaration
AI was primarily utilized for documentation purposes.
Components that developed or assisted using AI are detailed below.
They are also clearly marked in the code.
- `benchmark.py`:
    - Plotting mostly done using AI
    - Multithreading mostly was done using AI 

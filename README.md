# Job Shop Scheduling Problem (JSSP) - Genetic Algorithm

A Genetic Algorithm solver in Python for the Job Shop Scheduling Problem (JSSP) targeting makespan minimization.

## Features
- **Representation**: Operation-based permutation (job-ID repetition)
- **Selection**: Tournament selection
- **Crossover**: Precedence Preserving / Job-based Crossover (POX/JOX)
- **Mutation**: Swap mutation
- **Benchmark Data**: Lawrence benchmark instances in `data/` (e.g., `la01.txt`)

## Usage

Run the solver on the default benchmark instance:

```bash
python jobshop.py
```


# AutoStore Order-to-Station Assignment

This repository contains an AutoStore warehouse scheduling system focusing on order-to-station assignment using Constraint Programming (CP) and Heuristics. 

## Setup

The project relies on a Conda environment containing required dependencies like `docplex`, `cpoptimizer`, `pandas`, `plotly`, and `pytest`.

To set up the environment from the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate autostore
```

Alternative: use pip or uv with 'requirements.txt' and then install cplex manually.

```bash
uv venv --python 3.10
uv pip install -r requirements.txt
uv pip install cplex
```

It's also necessary to download the IBM CPLEX Studio (ideally the unrestricted version, available for free for academic use).
Then run this command in the project's environment.

```bash
docplex config --upgrade <PATH_TO_CPLEX_STUDIO>
```

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): problem summary, pipeline, map of every top-level module, workflows
- [docs/HEURISTICS.md](docs/HEURISTICS.md): the `heuristics/` package (SGC, AMA, RDI, partitioned RDI, CFSS, GBS), API, how to add a heuristic, equivalence harness
- [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md): known bugs and quirks

## Repository Structure

```text
.
├── heuristics/                       # constructive heuristics: run_heuristic(name, inst) -> HeuristicSolution
├── instance.py, datagen.py           # problem instance (+ cached attributes) and synthetic generator
├── cp_model.py                       # docplex CP Optimizer model (+ warm-start injection)
├── cp_lns_core.cpp, cpp_cp_model.py  # C++ CP model + Python wrapper (build: python setup.py mac|ubuntu)
├── matheuristic.py                   # CP-based LNS solver (+ neighborhood_selection, freezing_utils, matheuristic_plots)
├── precalculate_instances_for_lns.py # instance folders with heuristic warm start + CP baselines
├── decomposition.py                  # CP-based spatial / temporal / 2D / RL decompositions
├── rl_order_assignment.py            # RL order->station assignment
├── trainset_generator.py             # AMA dispatching-rule dataset
├── benchmark_models.py               # heuristic / RL benchmark
├── schedule_visualizer.py            # Plotly Gantt charts
├── docs/                             # documentation
├── tests/                            # pytest suite + tests/regression (equivalence vs. reference commit)
└── scripts/                          # one-off tools (migrate_heuristic_pickles.py, legacy scripts)
```

## How to Run

### Tests

```bash
python -m pytest tests/test_heuristics.py tests/test_instance.py -v
python tests/regression/equivalence.py --ref f391800   # heuristics identical to the pre-refactor commit
```

### Heuristics

```bash
python -m heuristics --list
python -m heuristics sgc --stations 1 --lanes 2 --orders 7 --skus 5 --seed 42 --pick 4 --horizon 2000
python -m heuristics rdi --orders 40 --param tiebreaker=sum_rt_asc --vis
python -m heuristics ama_two_phase --orders 40
```

```python
from datagen import generate_data
from heuristics import run_heuristic

inst = generate_data(num_stations=4, lanes_per_station=4, num_orders=160, num_skus=20000, seed=42, movecap=20)
sol = run_heuristic("lb_sgc", inst, horizon=10000)
print(sol.makespan, sol.validate(inst, 10000))
```

### CP model, LNS, benchmark

```bash
python precalculate_instances_for_lns.py --stations 4 --lanes 4 --orders 200 --skus 5000 --movecaps 20 \
       --gen-seeds 42 --cp-seeds 42 --heuristic lb_sgc
python matheuristic.py GEN42-4-4-5000-200-20 --backend cpp
python benchmark_models.py --num-instances 10 --methods all --skip-rl
```

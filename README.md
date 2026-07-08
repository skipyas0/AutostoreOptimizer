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

## Repository Structure

```text
.
├── analyze_heuristics/                 # Scripts for in-depth analysis of heuristic performance
├── benchmarking/                       # Benchmark scripts comparing heuristics and CP models
├── logs/                               # Directory containing output logs from various runs
├── results/                            # Directory for storing benchmark and solution output data
├── scripts/                            # Helper scripts (e.g., custom warm-starting, verification)
├── tests/                              # Pytest test suite covering all modules and edge cases
├── autostore_heuristic.py              # Baseline Sequential Greedy Construction (SGC) heuristic
├── cp_model.py                         # Core Constraint Programming (CP) Optimizer model
├── datagen.py                          # Data generation module for creating realistic benchmark instances
├── environment.yml                     # Conda environment definition with required dependencies
├── heuristic_ama_sgc.py                # Adaptive Multi-Attribute (AMA-SGC) heuristic
├── heuristic_ama_sgc_2phase.py         # Two-phase variant of the AMA-SGC heuristic
├── heuristic_cfss_sgc.py               # Cluster-First Schedule-Second (CFSS-SGC) heuristic
├── heuristic_gbs.py                    # Greedy Batching & Scheduling (GBS) baseline heuristic
├── heuristic_gbs_critical_path.py      # GBS variant focusing on critical path optimization
├── heuristic_gbs_max_sharing.py        # GBS variant focusing on maximizing bin sharing
├── heuristic_rdi_sgc.py                # Randomized Diversification Improvement (RDI-SGC) heuristic
├── heuristic_rdi_sgc_best_score.py     # RDI-SGC variant evaluating by best score
├── heuristic_rdi_sgc_sharing_degree.py # RDI-SGC variant evaluating by sharing degree
├── README.md                           # This documentation file
└── schedule_visualizer.py              # Tool for generating visual Gantt charts of schedules via Plotly
```

## How to Run

### Running Unit Tests

Run the test suite using `pytest` to ensure all modules work correctly:

```bash
python -m pytest tests/ -v
```

### Running Heuristics

You can run individual heuristics directly from the command line:

**Baseline SGC:**
```bash
python autostore_heuristic.py --stations 1 --lanes 2 --orders 7 --skus 5 --seed 42 --pick 4 --horizon 2000
```

**AMA-SGC Heuristic:**
```bash
python heuristic_ama_sgc.py --stations 1 --lanes 2 --orders 7 --skus 5 --seed 42 --pick 4 --horizon 2000
```

**CFSS-SGC Heuristic:**
```bash
python heuristic_cfss_sgc.py --stations 1 --lanes 2 --orders 7 --skus 5 --seed 42 --pick 4 --horizon 2000
```

### Running the CP Model

Execute the CP Optimizer model directly for a generated instance:
```bash
python cp_model.py --stations 1 --lanes 2 --orders 7 --skus 5 --seed 42 --pick 4 --timelimit 20 --horizon 2000
```

### Data Generation and Visualization

Generate new datasets or visualize outputs via integrated functionality in `datagen.py` and `schedule_visualizer.py`. Check their respective source files for usage examples.

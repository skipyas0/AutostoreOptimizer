import datetime
import json
import os
import re
import sys
import time
from contextlib import redirect_stdout

import numpy as np
from docplex.cp.solution import CpoModelSolution, CpoSolveResult

from autostore_heuristic import build_viz_handles, validate_solution
from cp_model import (
    ProgressCollector,
    build_model,
    extract_and_print_solution,
    inject_warmstart,
    validate_warmstart,
)
from datagen import generate_data
from heuristic_rdi_sgc import run_rdi_sgc
from neighborhoods import select_timeslice_neighborhood
from schedule_visualizer import plot_schedule, write_html


class Tee:
    """Write to multiple streams, flushing after each write."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


ts = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")  # noqa: DTZ005
path = f"experiments/{ts}"
os.mkdir(path)

config = {
    "stations": 6,
    "lanes": 6,
    "orders": 100,
    "symmetry_breaking": True,
    "skus": 30000,
    "movecap": 80,
    "seed": 42,
    "verbose": True,
    "collect_progress": True,
    "horizon": 10000,
    "alpha": 1.0,
    "beta": 1.0,
    "iters": 300,
    "stagnation_th": 50,
    "severity_step": 1,
    "iter_time_limit": 2.0,
    "baseline_ts_window": 100,
    "baseline_rand_pct": 0.05,
    "rrt_accept_th": 0.15,
    "use_whole_sol": True,
}

with open(f"{path}/config.json", "w+") as f:
    json.dump(config, f)

# Open log files
general_log = open(f"{path}/general_log.txt", "w")
matheur_log = open(f"{path}/matheur_log.txt", "w")
cp_log = open(f"{path}/cp_log.txt", "w")

instance = generate_data(
    num_stations=config["stations"],
    lanes_per_station=config["lanes"],
    num_orders=config["orders"],
    num_skus=config["skus"],
    seed=config["seed"],
    movecap=config["movecap"],
)


def validate_print(heur_sol, instance, config):
    violations = validate_solution(
        heur_sol, instance, horizon=config["horizon"], move_cap=config["movecap"]
    )

    if violations:
        print(f"VALIDATION FAILED ({len(violations)} violations)")
        for v in violations[:10]:
            print(f"  Violation: {v}")
    else:
        print("Validation PASSED")


# ========== GENERAL LOGS (heuristic, warmstart, comparisons) ==========
with redirect_stdout(Tee(sys.stdout, general_log)):
    print("=" * 60)
    print("GENERAL LOG - Heuristic & Warmstart")
    print("=" * 60)

    print("Running RDI-SGC Heuristic...")
    t_heur = time.perf_counter()
    heur_sol = run_rdi_sgc(
        instance,
        horizon=config["horizon"],
        move_cap=config["movecap"],
        ALPHA=config["alpha"],
        BETA=config["beta"],
    )

    print("\n=== RDI-SGC Heuristic Result ===")
    print(f"Feasible:    {heur_sol.feasible}")
    print(f"Makespan:    {heur_sol.makespan}")
    print(f"Total bin events (moves/2): {heur_sol.total_moves // 2}")
    print(f"Time:        {time.perf_counter() - t_heur:.4f}s")

    validate_print(heur_sol, instance, config)
    try:
        mock_sol, handles = build_viz_handles(heur_sol, instance)
        fig = plot_schedule(mock_sol, handles)
        html_file = f"{path}/Heur-RDI_solution.html"
        write_html(fig, html_file)
        print(f"\nWrote visualization to {html_file}")
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"[VIS] Skipped: {exc}")


def generate_freeze_constraints(
    mdl, handles, solution, severity, current_makespan, use_whole_sol=True
):
    # generate fix/optimize split
    if use_whole_sol:
        sp = solution
        if isinstance(solution, CpoSolveResult):
            sp = sp.get_solution()
    else:
        sp = mdl.create_empty_solution()
    # r = random.random()
    # if r < 0.33:
    #     print(" - Using: select_station_neighborhood")
    #     to_optimize = select_station_neighborhood(
    #         handles, solution, round(0.51 * severity)
    #     )
    # elif r < 0.66 or solution.get_objective_value() is None:
    #     print(" - Using: select_order_neighborhood")
    #     to_optimize = select_order_neighborhood(handles, 0.1 * severity)
    # else:
    #     print(" - Using: select_timeslice_neighborhood")
    to_optimize = select_timeslice_neighborhood(
        handles,
        solution,
        current_makespan,
        window_size=config["baseline_ts_window"] * severity,
        random_pct=config["baseline_rand_pct"] * severity,
    )
    # to_optimize = select_random_neighborhood(
    #     handles, solution, percent_to_free=0.01 * severity
    # )

    freeze_constraints = []

    for var_sol in solution.get_all_var_solutions():
        var = var_sol.get_var()

        # Check if the variable object itself is in the set
        if var in to_optimize:
            continue

        if "Interval" in var.type.name:
            # check if the interval is present in the current solution
            is_present = var_sol.is_present()

            if is_present:
                if not use_whole_sol:
                    sp.add_interval_var_solution(
                        var,
                        presence=True,
                        start=var_sol.get_start(),
                        end=var_sol.get_end(),
                        size=var_sol.get_size(),
                    )
                # freeze_constraints.append(mdl.presence_of(var) == 1)
                freeze_constraints.append(mdl.start_of(var) == var_sol.get_start())
                # freeze_constraints.append(mdl.end_of(var) == var_sol.get_end())
            else:
                # must stay absent
                if not use_whole_sol:
                    sp.add_interval_var_solution(var, presence=False)
                freeze_constraints.append(mdl.presence_of(var) == 0)

        elif any(t in var.type.name for t in ("Int", "Bool", "Float")):
            val = var_sol.get_value()  # record assigned numeric value
            freeze_constraints.append(var == val)
            if not use_whole_sol:
                sp.add_integer_var_solution(var, var_sol.get_value())

    return freeze_constraints, sp, to_optimize


mdl, handles = build_model(
    instance,
    rt_return=instance.rt_ret,
    add_symmetry_breaking=config["symmetry_breaking"],
    horizon=config["horizon"],
    move_cap=config["movecap"],
)

# ========== GENERAL LOGS (warmstart validation) ==========
with redirect_stdout(Tee(sys.stdout, general_log)):
    violations = validate_warmstart(heur_sol, heur_sol.pick_events, handles)
    if violations:
        print(f"Warmstart Violations Found ({len(violations)}):")
        for v in violations[:10]:
            print(f" - {v}")

sp = inject_warmstart(heur_sol, heur_sol.pick_events, mdl, handles)

curr_makespans = []
best_makespans = []
stagnation_increases = []
time_ticks = []
search_space_sizes = []
iter_statuses = {
    "Unknown": 0,
    "Optimal No improve": 0,
    "Optimal New best": 0,
    "Optimal Degradation": 0,
    "Feasible No improve": 0,
    "Feasible New best": 0,
    "Feasible Degradation": 0,
}

t_matheur = time.perf_counter()
MAX_ITER = config["iters"]

stagnation_count = 0
best_result = heur_sol.makespan
best_solution = sp

print(f"{sp=}, {sp.get_objective_value()=}")

num_vars = len(sp.get_all_var_solutions())
severity = 1
vns_step = config["severity_step"]
stagnation_th = config["stagnation_th"]

current_solution = best_solution
current_result = best_result

# ========== MATHEURISTIC / LNS LOOP LOGS ==========
with redirect_stdout(Tee(sys.stdout, matheur_log)):
    print("=" * 60)
    print("MATHEURISTIC / LNS LOOP")
    print("=" * 60)

    for i in range(MAX_ITER):
        curr_time = time.perf_counter() - t_matheur
        curr_makespans.append(current_result)
        best_makespans.append(best_result)
        time_ticks.append(curr_time)

        if stagnation_count >= stagnation_th:
            print(
                f"* Increasing neighborhood size: {severity} -> {vns_step + severity}\n"
            )
            severity += vns_step
            stagnation_count = 0
            stagnation_increases.append((i, curr_time))

        print(
            f"Iter {i}: Solving Partial CP Model | Best: {best_result} | Current: {current_result} | Severity: {severity}"
        )
        freeze_constraints, starting_point, to_optimize = generate_freeze_constraints(
            mdl,
            handles,
            current_solution,
            severity,
            current_result,
            use_whole_sol=config["use_whole_sol"],
        )
        mdl.set_starting_point(starting_point)
        mdl.add(freeze_constraints)
        # Solve the sub-problem
        print(f" - Optimizing {len(to_optimize)}/{num_vars} variables")

        sol = mdl.solve(
            TimeLimit=config["iter_time_limit"],
            LogVerbosity="Verbose",
            log_output=matheur_log,  # CP logs go directly to the matheur log file
            # Presolve="Off",  # Stop trying to simplify the 100k model
            # Workers=1
        )
        matheur_log.flush()  # ensure CP logs are written

        log_text = sol.get_solver_log()
        match = re.search(r"Log search space\s*:\s*([\d\.]+)", log_text)
        log_search_space = float(match.group(1)) if match else None
        search_space_sizes.append(log_search_space)
        print(f" - Log search space is {log_search_space}")

        if isinstance(sol, CpoModelSolution):
            print("WARN: sol is CpoModelSolution in loop")
            status = "Feasible"
        else:
            status = sol.get_solve_status().strip()

        print(f" - Partial optimization done, status: {status}")
        # ... Remove freeze constraints ...
        num_removed = mdl.remove(freeze_constraints)
        assert num_removed == len(freeze_constraints)

        improve_status = "No improve"
        accept = False
        if sol:
            # 2. ITERATIVE WARM START: Feed the CP solver's best result
            # back into the model for the next iteration.
            # docplex accepts the 'sol' object directly!

            if sol.get_objective_value() < current_result:
                accept = True
                print(
                    f" - Incumbent updated: {current_result} -> {sol.get_objective_value()}"
                )
                if sol.get_objective_value() < best_result:
                    improve_status = "New best"
                    print(
                        f" - New best found: {best_result} -> {sol.get_objective_value()}"
                    )
                    best_result = sol.get_objective_value()
                    best_solution = sol
                    stagnation_count = 0
            else:
                stagnation_count += 1
                if sol.get_objective_value() > current_result:
                    improve_status = "Degradation"

                print(
                    f" - Incumbent stagnates: {sol.get_objective_value()}>={current_result} for {stagnation_count} iterations"
                )
                # Record-to-record travel with threshold of 8%
                obj = sol.get_objective_value()
                gap = (obj - best_result) / best_result
                if obj and gap < config["rrt_accept_th"]:
                    accept = True
                    print(f" - Accepting worse with gap {100 * gap:0.2f}%")
                else:
                    print(f" - Rejecting worse with gap {100 * gap:0.2f}%")
        else:
            stagnation_count += 1
            print(
                f" - (CP NO SOLUTION) --> Incumbent stagnates at {best_result} for {stagnation_count} iterations"
            )

        if accept:
            current_solution = sol
            current_result = sol.get_objective_value()

        combined_status = (
            status if status == "Unknown" else f"{status} {improve_status}"
        )
        iter_statuses[combined_status] += 1

    mth_time = time.perf_counter() - t_matheur
    if best_solution:
        print(
            f"CP Solve Status: status: {'A solution' if isinstance(best_solution, CpoModelSolution) else best_solution.get_solve_status()}"
        )
        with open(f"{path}/matheur_sol.txt", "w+") as f, redirect_stdout(f):
            extract_and_print_solution(best_solution, handles)

        if plot_schedule:
            print("Exporting visualization...")
            fig = plot_schedule(best_solution, handles)
            html_file = f"{path}/CP-RDI-Matheur_solution.html"
            write_html(fig, html_file)
            print(f"\nWrote visualization to {html_file}")
    else:
        print("No solution found by Matheuristic CP.")

    print(f"Matheuristic total time: {mth_time:.4f}s")

# ========== STANDALONE CP LOGS ==========
with redirect_stdout(Tee(sys.stdout, cp_log)):
    print("=" * 60)
    print("STANDALONE CP SOLVER")
    print("=" * 60)

    # Run normal warmstared CP with the same warmstart with the same time it took the matheuristic
    mdl, handles = build_model(
        instance,
        rt_return=instance.rt_ret,
        add_symmetry_breaking=config["symmetry_breaking"],
        horizon=config["horizon"],
        move_cap=config["movecap"],
    )

    # Add history collection listener
    history_listener = ProgressCollector()
    mdl.add_solver_listener(history_listener)

    mdl.set_starting_point(sp)
    print(f"Solving CP Model with {mth_time}s time limit...")
    sol_cp = mdl.solve(
        TimeLimit=mth_time,
        LogVerbosity="Verbose",
        log_output=cp_log,  # CP logs go directly to the CP log file
        solve_with_search_next=True,
    )
    cp_log.flush()

    # Read the log text directly from the file on disk
    with open(f"{path}/cp_log.txt", "r") as f:
        log_text_file = f.read()

    match = re.search(r"Log search space\s*:\s*([\d\.]+)", log_text_file)
    log_search_space = float(match.group(1)) if match else None

    if sol_cp:
        print(f"CP Solve Status: {sol_cp.get_solve_status()}")
        with open(f"{path}/cp_sol.txt", "w+") as f, redirect_stdout(f):
            extract_and_print_solution(sol_cp, handles)
        if plot_schedule:
            print("Exporting visualization...")
            fig = plot_schedule(sol_cp, handles)
            html_file = f"{path}/CP-RDI_solution.html"
            write_html(fig, html_file)
            print(f"\nWrote visualization to {html_file}")
    else:
        print("No solution found by CP.")

# ========== GENERAL LOGS (final comparison) ==========
with redirect_stdout(Tee(sys.stdout, general_log)):
    print("=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"Matheuristic | {best_result} vs {sol_cp.get_objective_value()} | Normal CP")

# Close log files
general_log.close()
matheur_log.close()
cp_log.close()

import matplotlib.pyplot as plt
from matplotlib import ticker

plt.figure()
it = range(1, MAX_ITER + 1)
cp_time_ticks = [r["time"] for r in history_listener.records] + [mth_time]
cp_best_makespans = [r["best"] for r in history_listener.records] + [
    sol_cp.get_objective_value()
]

plt.plot(time_ticks, curr_makespans, color="orange", label="Current solution")
plt.plot(time_ticks, best_makespans, color="blue", label="Best solution")
plt.plot(cp_time_ticks, cp_best_makespans, color="black")

# plt.axhline(
#     y=heur_sol.makespan, color="grey", linestyle="-.", label="Heuristic solution"
# )
# plt.axhline(
# y=sol_cp.get_objective_value(), color="black", linestyle="-", label="CP solution"
# )

for s in stagnation_increases:
    # plot time of increases , label="Severity increase"
    plt.axvline(x=s[1], color="orchid", linestyle=":")

for i, t in zip(it, time_ticks):
    if i % 20 == 0:  # , label="LNS Iter"
        plt.axvline(x=t, color="palegreen", linestyle=":")
plt.title("Warmstarted Matheuristic performance vs Warmstarted CP")
plt.xlabel("Solve time")
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter("%d s"))
plt.ylabel("Makespan")
plt.legend(loc="upper left")


plt.savefig(
    f"{path}/matheuristic_vs_wscp.svg",
    format="svg",
    bbox_inches="tight",
)


plt.figure()
plt.plot(it, search_space_sizes, color="black")
for s in stagnation_increases:
    # plot time of increases , label="Severity increase"
    plt.axvline(x=s[0], color="thistle", linestyle=":")
plt.xlabel("Iteration")
plt.ylabel("Log Search Space Size")
plt.title(f"LNS Search Space Size, vs full {log_search_space}")
plt.savefig(
    f"{path}/search_space_sizes.svg",
    format="svg",
    bbox_inches="tight",
)


plt.figure()

bars = plt.bar(iter_statuses.keys(), iter_statuses.values())
plt.yscale("log")

plt.xticks(rotation=45, ha="right")

total_iters = sum(iter_statuses.values())
for bar in bars:
    height = bar.get_height()
    pct = (height / total_iters) * 100
    plt.annotate(
        f"{pct:.1f}%",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),  # 3 pt vertical offset
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.title("Iteration Results With Logarithmic Y-Axis")
plt.ylabel("Count (log scale)")
plt.savefig(
    f"{path}/iteration_results.svg",
    format="svg",
    bbox_inches="tight",
)
plt.figure()
iter_durations = np.diff(time_ticks)

# 1. Compute histogram counts and bin edges
counts, bin_edges, patches = plt.hist(
    iter_durations,
    bins=12,
    range=(0, 1.2 * config["iter_time_limit"]),
    log=True,
)

# 2. Fix log clipping so counts >= 1 are fully visible
plt.ylim(bottom=0.8)

# 3. Annotate percentages above bars
total_durations = len(iter_durations)
for count, patch in zip(counts, patches):
    if count > 0:
        pct = (count / total_durations) * 100
        height = patch.get_height()
        plt.annotate(
            f"{pct:.1f}%",
            xy=(patch.get_x() + patch.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

# 4. Format X-axis and labels
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter("%g s"))
plt.xlabel("Duration (seconds)")
plt.ylabel("Frequency (log scale)")
plt.title("Iteration Durations With Logarithmic Y-Axis")

plt.tight_layout()
plt.savefig(
    f"{path}/iteration_durations.svg",
    format="svg",
    bbox_inches="tight",
)

# 5. Build and save the JSON table
histogram_data = []
for i in range(len(counts)):
    histogram_data.append(
        {
            "bin_index": i,
            "bin_start_sec": float(bin_edges[i]),
            "bin_end_sec": float(bin_edges[i + 1]),
            "count": int(counts[i]),
            "percentage": round(float((counts[i] / total_durations) * 100), 2),
        }
    )

with open(f"{path}/iteration_durations_bins.json", "w") as f:
    json.dump(
        {"histogram": histogram_data, "durations": list(iter_durations)}, f, indent=4
    )

import random
import time

from autostore_heuristic import validate_solution
from cp_model import (
    build_model,
    extract_and_print_solution,
    inject_warmstart,
    validate_warmstart,
)
from datagen import generate_data
from heuristic_rdi_sgc import run_rdi_sgc
from schedule_visualizer import plot_schedule, write_html

config = {
    "stations": 1,
    "lanes": 4,
    "orders": 20,
    "timelimit": 200,
    "symmetry_breaking": True,
    "skus": 20000,
    "movecap": 20,
    "seed": 42,
    "verbose": True,
    "collect_progress": True,
    "horizon": 10000,
    "alpha": 1.0,
    "beta": 1.0,
}

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


def select_station_neighborhood(handles, solution, num_stations=2):
    unfrozen_vars = set()

    S = handles["S"]
    O = handles["O"]
    L = handles["L"]
    orders_req = handles["orders_req"]
    U = handles["U"]

    # 1. Pick a random subset of stations to completely clear out
    free_stations = set(random.sample(S, min(num_stations, len(S))))

    # 2. Find ALL orders that are currently assigned to these stations
    free_orders = set()
    for o in O:
        for s in free_stations:
            var = handles["I_os"][(o, s)]
            var_sol = solution.get_var_solution(var)

            # If the order is present at the free station, add it to our list
            if var_sol is not None and var_sol.is_present():
                free_orders.add(o)
                break

    # 3. Unfreeze EVERYTHING for the selected orders (across all stations)
    # This allows them to change stations, change lanes, and change times
    for o in free_orders:
        for s in S:
            unfrozen_vars.add(handles["I_os"][(o, s)])
            for ln in L:
                unfrozen_vars.add(handles["I_os_lane"][(o, s, ln)])
            for k in orders_req[o]:
                unfrozen_vars.add(handles["C"][(o, k, s)])
                for e in range(U[k]):
                    if (o, s, k, e) in handles["P"]:
                        unfrozen_vars.add(handles["P"][(o, s, k, e)])

    # 4. Unfreeze ALL Bin variables at the selected stations
    # Because all orders at these stations are free, the bins are completely
    # detached from the frozen schedule and can be freely rearranged.
    for s in free_stations:
        for k in handles["K"]:
            if U.get(k, 0) > 0:
                for e in range(U[k]):
                    unfrozen_vars.add(handles["F"][(s, k, e)])
                    unfrozen_vars.add(handles["R"][(s, k, e)])
                    unfrozen_vars.add(handles["B"][(s, k, e)])
                    unfrozen_vars.add(handles["Block"][(s, k, e)])

    return unfrozen_vars


def select_order_neighborhood(handles, percent_to_free=0.15):
    unfrozen_vars = set()

    O = handles["O"]
    S = handles["S"]
    L = handles["L"]
    orders_req = handles["orders_req"]

    # 1. Select a random subset of orders
    num_orders = max(1, int(len(O) * percent_to_free))
    free_orders = set(random.sample(O, num_orders))

    # 2. Find which SKUs those orders need
    free_skus = set()
    for o in free_orders:
        free_skus.update(orders_req[o])

    # 3. Unfreeze all Order-level variables for the selected orders
    for o in free_orders:
        for s in S:
            unfrozen_vars.add(handles["I_os"][(o, s)])
            for ln in L:
                unfrozen_vars.add(handles["I_os_lane"][(o, s, ln)])
            for k in orders_req[o]:
                unfrozen_vars.add(handles["C"][(o, k, s)])

                # Unfreeze the Picks associated with this order
                for e in range(handles["U"][k]):
                    unfrozen_vars.add(handles["P"][(o, s, k, e)])

    # 4. Unfreeze ALL Bin cycle variables (F, R, B, Block) for the affected SKUs
    # This allows the solver to re-route bins across different stations
    # to serve the newly freed orders optimally.
    for k in free_skus:
        for s in S:
            for e in range(handles["U"][k]):
                unfrozen_vars.add(handles["F"][(s, k, e)])
                unfrozen_vars.add(handles["R"][(s, k, e)])
                unfrozen_vars.add(handles["B"][(s, k, e)])
                unfrozen_vars.add(handles["Block"][(s, k, e)])

    return unfrozen_vars


def select_timeslice_neighborhood(handles, solution, window_size=1000):
    unfrozen_vars = set()
    max_time = int(solution.get_objective_value())

    # Pick a time window
    start_time = random.randint(0, max(0, max_time - window_size))
    end_time = start_time + window_size

    # Iterate through the solution and unfreeze any variable that touches this window
    for var_sol in solution.get_all_var_solutions():
        var = var_sol.get_var()

        if "Interval" in var.type.name:
            if var_sol.is_present():
                if var_sol.get_start() <= end_time and var_sol.get_end() >= start_time:
                    unfrozen_vars.add(var)
            else:
                # Always unfreeze a portion of absent variables so the solver
                # can try to insert them into the slack we are creating
                if random.random() < 0.2:
                    unfrozen_vars.add(var)

    # THE IRON RULE OF COUPLED INTERVALS:
    # If the time-slice caught a Fetch (F), we MUST manually add the paired
    # Return (R), Bin (B), and Block variables to prevent gridlock.
    # (You would add a quick loop here iterating through unfrozen_vars,
    # parsing the string name or using handles to pull the sibling variables).

    return unfrozen_vars


def generate_freeze_constraints(mdl, handles, solution, severity):
    # generate fix/optimize split
    sp = mdl.create_empty_solution()
    r = random.random()
    if r < 0.33:
        print(" - Using: select_station_neighborhood")
        to_optimize = select_station_neighborhood(
            handles, solution, round(0.51 * severity)
        )
    elif r < 0.66 or solution.get_objective_value() is None:
        print(" - Using: select_order_neighborhood")
        to_optimize = select_order_neighborhood(handles, 0.1 * severity)
    else:
        print(" - Using: select_timeslice_neighborhood")
        to_optimize = select_timeslice_neighborhood(handles, solution, 50 * severity)

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
                sp.add_interval_var_solution(
                    var,
                    presence=True,
                    start=var_sol.get_start(),
                    end=var_sol.get_end(),
                    size=var_sol.get_size(),
                )
                freeze_constraints.append(mdl.presence_of(var) == 1)
                freeze_constraints.append(mdl.start_of(var) == var_sol.get_start())
                freeze_constraints.append(mdl.end_of(var) == var_sol.get_end())
            else:
                # must stay absent
                sp.add_interval_var_solution(var, presence=False)
                freeze_constraints.append(mdl.presence_of(var) == 0)

        elif any(t in var.type.name for t in ("Int", "Bool", "Float")):
            val = var_sol.get_value()  # record assigned numeric value
            freeze_constraints.append(var == val)
            sp.add_integer_var_solution(var, var_sol.get_value())

    return freeze_constraints, sp, to_optimize


mdl, handles = build_model(
    instance,
    rt_return=instance.rt_ret,
    add_symmetry_breaking=config["symmetry_breaking"],
    horizon=config["horizon"],
    move_cap=config["movecap"],
)


violations = validate_warmstart(heur_sol, heur_sol.pick_events, handles)
if violations:
    print(f"Warmstart Violations Found ({len(violations)}):")
    for v in violations[:10]:
        print(f" - {v}")

sp = inject_warmstart(heur_sol, heur_sol.pick_events, mdl, handles)

curr_makespans = []
best_makespans = []
stagnation_increases = []
t_matheur = time.perf_counter()
MAX_ITER = 200

stagnation_count = 0
best_result = heur_sol.makespan
best_solution = sp

print(f"{sp=}, {sp.get_objective_value()=}")

num_vars = len(sp.get_all_var_solutions())
severity = 1
vns_step = 1
stagnation_th = 50

current_solution = best_solution
current_result = best_result
for i in range(MAX_ITER):
    curr_makespans.append(current_result)
    best_makespans.append(best_result)

    if stagnation_count >= stagnation_th:
        print(f"* Increasing neighborhood size: {severity} -> {vns_step + severity}\n")
        severity += vns_step
        stagnation_count = 0
        stagnation_increases.append(i)

    print(
        f"Iter {i}: Solving Partial CP Model | Best: {best_result} | Current: {current_result} | Severity: {severity}"
    )
    freeze_constraints, starting_point, to_optimize = generate_freeze_constraints(
        mdl, handles, current_solution, severity
    )
    mdl.set_starting_point(starting_point)
    mdl.add(freeze_constraints)
    # Solve the sub-problem
    print(f" - Optimizing {len(to_optimize)}/{num_vars} variables")
    sol = mdl.solve(
        FailLimit=100000,
        LogVerbosity="Quiet",
        # Presolve="Off",  # Stop trying to simplify the 100k model
        # Workers=1
    )

    print(f" - Partial optimization done, status: {sol.get_solve_status()}")
    # ... Remove freeze constraints ...
    num_removed = mdl.remove(freeze_constraints)
    assert num_removed == len(freeze_constraints)

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
                print(
                    f" - New best found: {best_result} -> {sol.get_objective_value()}"
                )
                best_result = sol.get_objective_value()
                best_solution = sol
                stagnation_count = 0
        else:
            stagnation_count += 1
            print(
                f" - Incumbent stagnates: {sol.get_objective_value()}>={current_result} for {stagnation_count} iterations"
            )
            # Record-to-record travel with threshold of 8%
            obj = sol.get_objective_value()
            gap = (obj - best_result) / best_result
            if obj and gap < 0.15:
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

mth_time = time.perf_counter() - t_matheur
if best_solution:
    print(f"CP Solve Status: {best_solution.get_solve_status()}")
    extract_and_print_solution(best_solution, handles)
    if plot_schedule:
        print("Exporting visualization...")
        fig = plot_schedule(best_solution, handles)
        html_file = "./CP-RDI-mat_solution.html"
        write_html(fig, html_file)
        print(f"\nWrote visualization to {html_file}")
else:
    print("No solution found by Matheuristic CP.")

print(mth_time)


# Run normal warmstared CP with the same warmstart with the same time it took the matheuristic


mdl, handles = build_model(
    instance,
    rt_return=instance.rt_ret,
    add_symmetry_breaking=config["symmetry_breaking"],
    horizon=config["horizon"],
    move_cap=config["movecap"],
)
mdl.set_starting_point(sp)
print(f"Solving CP Model with {config['timelimit']}s time limit...")
sol_cp = mdl.solve(TimeLimit=mth_time, LogVerbosity="Terse")

if sol_cp:
    print(f"CP Solve Status: {sol_cp.get_solve_status()}")
    extract_and_print_solution(sol_cp, handles)
    if plot_schedule:
        print("Exporting visualization...")
        fig = plot_schedule(sol_cp, handles)
        html_file = "./CP-RDI_solution.html"
        write_html(fig, html_file)
        print(f"\nWrote visualization to {html_file}")
else:
    print("No solution found by CP.")

print(f"Matheuristic | {best_result} vs {sol_cp.get_objective_value()} | Normal CP")

import matplotlib.pyplot as plt

plt.figure()
it = range(1, MAX_ITER + 1)
plt.plot(it, curr_makespans, color="orange", label="Current solution")
plt.plot(it, best_makespans, color="blue", label="Best solution")
plt.axhline(
    y=heur_sol.makespan, color="grey", linestyle="-.", label="Heuristic solution"
)
plt.axhline(
    y=sol_cp.get_objective_value(), color="black", linestyle="-", label="CP solution"
)
for s in stagnation_increases:
    plt.axvline(x=s, color="thistle", linestyle=":", label="Severity increase")
plt.title("Warmstarted Matheuristic performance vs Warmstarted CP")
plt.xlabel("Iteration")
plt.ylabel("Makespan")
plt.legend(loc="upper left")
plt.savefig("matheuristic_vs_wscp.svg", format="svg", bbox_inches="tight")

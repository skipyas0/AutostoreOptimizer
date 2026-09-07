from autostore_heuristic import BinEvent, Solution


def cp_sol_to_solution(sol, handles):
    def iv_present(x):
        if x is None:
            return False
        if hasattr(sol, "BooleanValue"):
            return sol.BooleanValue(x.pres)
        vs = sol.get_var_solution(x)
        return (vs is not None) and vs.is_present()

    def iv_start(x):
        if hasattr(sol, "Value"):
            return sol.Value(x.start)
        return sol.get_var_solution(x).get_start()

    def iv_end(x):
        if hasattr(sol, "Value"):
            return sol.Value(x.end)
        return sol.get_var_solution(x).get_end()

    I_os_lane = handles["I_os_lane"]
    I_os = handles["I_os"]
    P = handles["P"]
    F = handles["F"]
    R = handles["R"]
    B = handles["B"]
    U = handles["U"]
    orders_req = handles["orders_req"]
    S, L, K, O = handles["S"], handles["L"], handles["K"], handles["O"]

    makespan = 0
    for o in O:
        ends = [iv_end(I_os[(o, s)]) for s in S if iv_present(I_os[(o, s)])]
        if ends:
            makespan = max(makespan, max(ends))

    order_assignments = {}
    for o in O:
        s_sel = next((s for s in S if iv_present(I_os[(o, s)])), None)
        if s_sel is not None:
            ln_sel = next(ln for ln in L if iv_present(I_os_lane[(o, s_sel, ln)]))
            st, en = iv_start(I_os[(o, s_sel)]), iv_end(I_os[(o, s_sel)])
            order_assignments[o] = (s_sel, ln_sel, st, en)

    bin_events = {s: [] for s in S}
    total_moves = 0

    pick_events = {}

    for s in S:
        for k in K:
            Uk = U[k]
            for e in range(Uk):
                if (s, k, e) in B and iv_present(B[(s, k, e)]):
                    total_moves += 2

                    fs, fe = iv_start(F[(s, k, e)]), iv_end(F[(s, k, e)])
                    bs, be = iv_start(B[(s, k, e)]), iv_end(B[(s, k, e)])
                    rs, re = iv_start(R[(s, k, e)]), iv_end(R[(s, k, e)])

                    orders_served = []
                    for o in O:
                        if (
                            k in orders_req[o]
                            and (o, s, k, e) in P
                            and iv_present(P[(o, s, k, e)])
                        ):
                            ps, pe = (
                                iv_start(P[(o, s, k, e)]),
                                iv_end(P[(o, s, k, e)]),
                            )
                            orders_served.append(o)
                            pick_events[(o, s, k)] = (ps, pe)

                    be_obj = BinEvent(
                        sku=k,
                        copy_id=e,
                        fetch_start=fs,
                        fetch_end=fe,
                        presence_start=bs,
                        presence_end=be,
                        return_start=rs,
                        return_end=re,
                        orders_served=orders_served,
                    )
                    bin_events[s].append(be_obj)

    return Solution(
        order_assignments=order_assignments,
        bin_events=bin_events,
        makespan=makespan,
        total_moves=total_moves,
        feasible=True,
        pick_events=pick_events,
    )


def sort_variables(mdl, sp, backend="docplex", handles=None):
    if backend == "cpp":

        def get_start_time(var):
            var_sol = sp.get_var_solution(var)
            if var_sol is not None and var_sol.is_present():
                return var_sol.get_start()
            return float("inf")

        interval_vars = list(handles["all_intervals_flat"].values())
        sorted_vars = sorted(
            interval_vars, key=lambda v: (get_start_time(v), v.get_name())
        )
        var_to_idx = {var: idx for idx, var in enumerate(sorted_vars)}
        return var_to_idx, len(var_to_idx)

    def get_start_time(var):
        var_sol = sp.get_var_solution(var)
        if var_sol is not None and var_sol.is_present():
            return var_sol.get_start()
        # Push absent variables to the very top (or bottom) of the y-axis
        return float("inf")

    # 1. Filter only interval variables
    interval_vars = [
        var for var in mdl.get_all_variables() if "Interval" in var.type.name
    ]

    # 2. Sort chronologically by start time in the warmstart (sp),
    # using the variable name as a secondary tie-breaker for stability.
    sorted_vars = sorted(interval_vars, key=lambda v: (get_start_time(v), v.get_name()))

    # 3. Create the mapping based on the sorted order
    var_to_idx = {var: idx for idx, var in enumerate(sorted_vars)}
    num_variables = len(var_to_idx)
    return var_to_idx, num_variables


class MockIntervalVar:
    def __init__(self, name, present=False, start=None, end=None, size=None):
        self.name = name
        self.present = present
        self._start = start
        self._end = end
        self._size = size

    def get_name(self):
        return self.name

    def is_present(self):
        return self.present

    def get_start(self):
        return self._start

    def get_end(self):
        return self._end

    def get_size(self):
        return self._size

    # Adding for compatibility with some scripts that might call get_var()
    def get_var(self):
        return self

    def get_value(self):
        return self


class CppSolveResult:
    def __init__(self, result_dict):
        self.status = result_dict.get("status", "Unknown")
        self.objective = result_dict.get("objective", None)
        self.var_sols = result_dict.get("var_solutions", {})
        self.total_time = result_dict.get("TotalTime", 0.0)
        self.extraction_time = result_dict.get("ExtractionTime", 0.0)
        self.solve_time = result_dict.get("SolveTime", 0.0)

    def get_info(self, key):
        if key == "TotalTime":
            return self.total_time
        if key == "ExtractionTime":
            return self.extraction_time
        if key == "SolveTime":
            return self.solve_time
        return None

    def get_solve_time(self):
        return self.solve_time

    def get_solve_status(self):
        return self.status

    def get_objective_value(self):
        return self.objective

    def get_var_solution(self, var):
        if hasattr(var, "get_name"):
            name = var.get_name()
        else:
            name = var

        if name in self.var_sols:
            vs = self.var_sols[name]
            return MockIntervalVar(
                name=name,
                present=vs["present"],
                start=vs.get("start"),
                end=vs.get("end"),
                size=vs.get("size"),
            )
        return None

    def get_all_var_solutions(self):
        sols = []
        for name, vs in self.var_sols.items():
            sols.append(
                MockIntervalVar(
                    name=name,
                    present=vs["present"],
                    start=vs.get("start"),
                    end=vs.get("end"),
                    size=vs.get("size"),
                )
            )
        return sols

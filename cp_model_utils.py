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


def build_solve_dict(sres):
    sol_dict = {}
    for var_sol in sres.get_all_var_solutions():
        val = var_sol.get_value()
        if hasattr(val, "is_present"):
            sol_dict[var_sol.get_name()] = {
                "present": val.is_present(),
                "start": val.get_start() if val.is_present() else None,
                "end": val.get_end() if val.is_present() else None,
            }
        else:
            sol_dict[var_sol.get_name()] = val
    return sol_dict


def get_fleet_utilization_timeseries(solution, handles, makespan):
    """Returns a dense list of length `makespan + 1` where index `t`

    corresponds to the number of active fleet moves during time step `t`.
    """
    # Array of size makespan + 2 to safely record differences at `makespan`
    diff = [0] * (int(makespan) + 2)

    def register_interval(iv):
        if iv is None:
            return
        if hasattr(solution, "get_var_solution"):
            sol = solution.get_var_solution(iv)
        else:
            sol = solution.get(iv)

        if sol and sol.is_present():
            st = sol.get_start()
            en = sol.get_end()
            # Only record if the interval starts within the tracked horizon
            if st <= makespan:
                diff[st] += 1
                if en <= makespan:
                    diff[en] -= 1

    # Record +1 at start and -1 at end for all active F and R intervals[cite: 1]
    for f_var in handles["F"].values():
        register_interval(f_var)

    for r_var in handles["R"].values():
        register_interval(r_var)

    # Compute prefix sums to get dense utilization per discrete time unit
    utilization = [0] * (int(makespan) + 1)
    current_moves = 0
    for t in range(int(makespan) + 1):
        current_moves += diff[t]
        utilization[t] = current_moves

    return utilization


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


def canonicalize_cpo(cpo_path: str, output_path: str | None = None) -> str:
    """Parses a .cpo file (produced either by docplex CpoModel.export_model()

    or CP Optimizer C++ cp.dumpModel()), normalizes variables, expressions,
    intermediate Concert aliases, and constraints into a canonical, sorted
    CPO text format.

    If output_path is provided, writes the canonical CPO model to that file.
    Returns the canonical CPO string.
    """
    import re

    from docplex.cp.cpo.cpo_parser import CpoParser
    from docplex.cp.expression import CpoIntervalVar, CpoSequenceVar, CpoValue

    p = CpoParser().parse(cpo_path)

    # 1. Clear intermediate alias names on expressions (e.g. from Concert dumpModel)
    def clear_names(expr):
        if (
            hasattr(expr, "name")
            and expr.name
            and re.match(
                r"^(IntervalPresence|IntervalExpr|VarCumulAtom)_\d+$", str(expr.name)
            )
        ):
            expr.name = None
        if hasattr(expr, "children"):
            for c in expr.children:
                clear_names(c)

    for e, loc in p.get_all_expressions():
        clear_names(e)

    # 2. Extract and canonicalize variable declarations
    var_decls = []
    for v in p.get_all_variables():
        if not isinstance(v, CpoIntervalVar):
            continue
        vname = v.get_name()
        v_str = f'"{vname}" = intervalVar(optional'
        if hasattr(v, "get_size") and v.get_size() is not None:
            sz = v.get_size()
            if isinstance(sz, int) or (isinstance(sz, tuple) and sz[0] == sz[1]):
                val = sz if isinstance(sz, int) else sz[0]
                v_str += f", size={val}"
        v_str += ");"
        var_decls.append(v_str)
    var_decls.sort()

    # 3. Canonical expression formatter
    def unwrap_list(val):
        if isinstance(val, (list, tuple)):
            res = []
            for x in val:
                res.extend(unwrap_list(x))
            return res
        if isinstance(val, CpoValue) and isinstance(val.value, (list, tuple)):
            return unwrap_list(val.value)
        if isinstance(val, CpoSequenceVar):
            return unwrap_list(val.children)
        return [val]

    def canonical_expr_str(e):
        if isinstance(e, CpoIntervalVar):
            return f'"{e.get_name()}"'
        if isinstance(e, CpoValue):
            if isinstance(e.value, (list, tuple)):
                items = sorted(canonical_expr_str(x) for x in e.value)
                return "[" + ", ".join(items) + "]"
            return str(e.value)

        op = getattr(e, "operation", None)
        op_name = op.cpo_name if op else None

        # Unwrap sequenceVar in noOverlap
        if op_name == "noOverlap":
            child = e.children[0]
            raw_items = unwrap_list(child)
            items = sorted(canonical_expr_str(x) for x in raw_items)
            return "noOverlap([" + ", ".join(items) + "])"

        # Unwrap alternatives / span
        if op_name in ("alternative", "span"):
            head = canonical_expr_str(e.children[0])
            cand_list = unwrap_list(e.children[1])
            items = sorted(canonical_expr_str(x) for x in cand_list)
            return f"{op_name}({head}, [" + ", ".join(items) + "])"

        # Filter pulse(..., 0) in alwaysIn
        if op_name == "alwaysIn":
            cumul = e.children[0]
            start_val = canonical_expr_str(e.children[1])
            end_val = canonical_expr_str(e.children[2])
            min_val = canonical_expr_str(e.children[3])
            max_val = canonical_expr_str(e.children[4])

            pulses = []

            def collect_pulses(node):
                nop = getattr(node, "operation", None)
                if nop and nop.cpo_name in ("sum", "plus"):
                    for ch in node.children:
                        for x in unwrap_list(ch):
                            collect_pulses(x)
                elif nop and nop.cpo_name == "pulse":
                    if len(node.children) >= 2:
                        h_node = node.children[1]
                        h_val = (
                            h_node.value
                            if isinstance(h_node, CpoValue)
                            else str(h_node)
                        )
                        first_arg = canonical_expr_str(node.children[0])
                        if (
                            str(h_val) != "0"
                            and "-4503599" not in first_arg
                            and "intervalmin" not in first_arg
                        ):
                            pulses.append(f"pulse({first_arg}, {h_val})")
                else:
                    s = str(node)
                    if (
                        "pulse(intervalmin, intervalmax, 0)" not in s
                        and "-4503599" not in s
                        and s != "0"
                    ):
                        pulses.append(canonical_expr_str(node))

            collect_pulses(cumul)
            pulses.sort()
            return (
                "alwaysIn(sum(["
                + ", ".join(pulses)
                + f"]), {start_val}, {end_val}, {min_val}, {max_val})"
            )

        # Commutative equal
        if op_name == "equal":
            left = canonical_expr_str(e.children[0])
            right = canonical_expr_str(e.children[1])
            if (
                left.isdigit()
                and not right.isdigit()
                or not (right.isdigit() and not left.isdigit())
                and left > right
            ):
                left, right = right, left
            return f"{left} == {right}"

        # Commutative sum / plus in linear expressions
        if op_name in ("sum", "plus"):
            parts = []

            def collect_sum(node):
                nop = getattr(node, "operation", None)
                if nop and nop.cpo_name in ("sum", "plus"):
                    for ch in node.children:
                        for x in unwrap_list(ch):
                            collect_sum(x)
                else:
                    parts.append(canonical_expr_str(node))

            collect_sum(e)
            parts.sort()
            return " + ".join(parts)

        # Objective or nested max
        if op_name in ("minimize", "maximize"):
            leaves = []

            def collect_max(node):
                nop = getattr(node, "operation", None)
                if nop and nop.cpo_name == "max":
                    for ch in node.children:
                        for x in unwrap_list(ch):
                            collect_max(x)
                else:
                    leaves.append(canonical_expr_str(node))

            collect_max(e.children[0])
            leaves.sort()
            return f"{op_name}(max([" + ", ".join(leaves) + "]))"

        if op_name == "max":
            leaves = []

            def collect_max(node):
                nop = getattr(node, "operation", None)
                if nop and nop.cpo_name == "max":
                    for ch in node.children:
                        for x in unwrap_list(ch):
                            collect_max(x)
                else:
                    leaves.append(canonical_expr_str(node))

            collect_max(e)
            leaves.sort()
            if len(leaves) == 1:
                return leaves[0]
            return "max([" + ", ".join(leaves) + "])"

        # If-then
        if op_name == "ifThen":
            cond = canonical_expr_str(e.children[0])
            then_expr = canonical_expr_str(e.children[1])
            return f"ifThen({cond}, {then_expr})"

        # Default recursive
        if hasattr(e, "children") and e.children:
            args = [canonical_expr_str(c) for c in e.children]
            op_sym = getattr(op, "keyword", None)
            if op_sym and len(args) == 2 and op_sym not in ("==",):
                return f"{args[0]} {op_sym} {args[1]}"
            cpo_fn = getattr(op, "cpo_name", str(op))
            return f"{cpo_fn}(" + ", ".join(args) + ")"
        return str(e)

    constraints = []
    objective = None
    for e, loc in p.get_all_expressions():
        if e.__class__.__name__ == "CpoFunctionCall":
            op = getattr(e, "operation", None)
            if op and op.cpo_name in ("minimize", "maximize"):
                objective = canonical_expr_str(e) + ";"
            else:
                constraints.append(canonical_expr_str(e) + ";")
    constraints.sort()

    lines = ["//--- Variables ---"] + var_decls
    if objective:
        lines += ["", "//--- Objective ---", objective]
    lines += ["", "//--- Constraints ---"] + constraints
    content = "\n".join(lines) + "\n"

    if output_path:
        with open(output_path, "w") as f:
            f.write(content)

    return content


def compare_cpo_files(
    py_cpo_path: str,
    cpp_cpo_path: str,
    py_canonical_path: str | None = None,
    cpp_canonical_path: str | None = None,
) -> tuple[bool, list[str]]:
    """Canonicalizes both .cpo files and performs a line-by-line unified diff.

    Returns (is_identical, diff_lines).
    """
    import difflib

    c_py = canonicalize_cpo(py_cpo_path, py_canonical_path)
    c_cpp = canonicalize_cpo(cpp_cpo_path, cpp_canonical_path)

    diff = list(
        difflib.unified_diff(
            c_py.splitlines(keepends=True),
            c_cpp.splitlines(keepends=True),
            fromfile="py_canonical.cpo",
            tofile="cpp_canonical.cpo",
        )
    )
    return len(diff) == 0, diff

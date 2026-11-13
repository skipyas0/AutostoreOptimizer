"""Compare every method under an experiment root: LNS configurations, CP, heuristics.

Writes into OUT:
  tables/   CSV + Markdown + LaTeX (+ PNG with --png-tables)
  figures/  PDF + PNG
  data/     per-run and per-iteration CSVs for further analysis
  README.md index with the headline numbers

Figures and tables that compare methods come in two variants:
  *_full     all LNS configs, CP and all heuristics
  *_reduced  best LNS config + the baselines (default: CP and RDI)
"""

import os
import warnings

import numpy as np
import pandas as pd

from .. import metrics, plots, style
from ..configs import assign_lns_methods, config_params_table, config_table
from ..data import instance_dirs, load_root
from ..solutions import METRICS, solution_metrics
from ..tables import fmt_num, mean_std, to_markdown, write_table


OBJ_LABELS = {"makespan": "makespan", "total_flow_time": "flow time",
              "bin_fetches": "bin fetches", "idle_time": "idle time",
              "max_picks": "max picks", "sum_makespans": "sum of station makespans",
              "total_wages": "wages"}


def obj_label(obj):
    return OBJ_LABELS.get(obj, obj)


def _fmt(df, digits):
    """Round float columns for display (the CSV keeps these rounded values too)."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].map(lambda x: fmt_num(x, f"{{:.{digits}f}}"))
    return out


def _gap_cell(v):
    """mean ± std, flagged when some runs have no solution yet (they are excluded)."""
    return mean_std(v) if v.notna().all() else \
        f"{mean_std(v)} ({v.isna().sum()}/{len(v)} w/o sol.)"


class Comparison:
    def __init__(self, root, out, time_points=(60, 300, 900), reduced_baselines=("CP", "RDI"),
                 pattern="GEN*", png_tables=False, select=None):
        self.root, self.out, self.pattern = root, out, pattern
        self.time_points = list(time_points)
        self.reduced_baselines = list(reduced_baselines)
        self.png_tables = png_tables
        self.tab = os.path.join(out, "tables")
        self.fig = os.path.join(out, "figures")
        self.dat = os.path.join(out, "data")

        d = load_root(root, pattern)
        self.instances, self.sizes = d.instances, d.size_order
        # Ids come from all configs, so LNS-k means the same config in every report.
        self.cfg = config_table(d.experiments, select=select)
        runs, self.iters = assign_lns_methods(d.runs, d.iters, self.cfg)
        # Best known solution from everything loaded, before filtering, so gaps are
        # comparable between reports on the same root.
        runs = metrics.add_reference(runs, self.instances)
        if select:
            keep = runs.family.ne("lns") | runs.method.isin(select)
            runs = runs[keep]
            if len(self.iters):
                self.iters = self.iters[self.iters.method.isin(select)]
        self.runs = metrics.at_times(runs, self.time_points)
        self.time_limit = max(self.time_points)
        self._methods()

    # ------------------------------------------------------------------ #
    def _methods(self):
        r = self.runs
        lns = list(self.cfg.method)
        heur = (r[r.family == "heuristic"].groupby("method").gap_final.mean()
                .sort_values().index.tolist())
        start = r[(r.family == "heuristic") & (r.is_start == True)].method.unique()  # noqa: E712
        self.start_name = start[0] if len(start) else None
        self.lns, self.heuristics = lns, heur
        self.methods = lns + (["CP"] if (r.method == "CP").any() else []) + heur

        look = {}
        cfg_label = self.cfg.set_index("method").label
        for i, m in enumerate(lns):
            look[m] = dict(family="lns", color=style.CATEGORICAL[i % 8],
                           label=f"{m}: {cfg_label[m]}")
        look["CP"] = dict(family="cp", color=style.CP_COLOR, label="CP (full model)")
        for m in heur:
            lab = f"{m} (LNS start)" if m == self.start_name else m
            look[m] = dict(family="heuristic", color=style.HEURISTIC_COLOR, label=lab)
        self.look = look

        # best LNS config: lowest mean final gap over instances
        inst_gap = metrics.instance_means(r[r.method.isin(lns)], "gap_final")
        self.best_lns = inst_gap.mean().idxmin() if lns else None
        missing = [b for b in self.reduced_baselines if b not in self.methods]
        if missing:
            warnings.warn(f"reduced baselines not found: {missing}")
        self.reduced = ([self.best_lns] if self.best_lns else []) + \
            [b for b in self.reduced_baselines if b in self.methods]

    def variants(self):
        return [("full", self.methods), ("reduced", self.reduced)]

    def _write(self, df, name, caption):
        write_table(df, self.tab, name, caption, png=self.png_tables)

    # ------------------------------------------------------------------ #
    # Tables
    # ------------------------------------------------------------------ #
    def table_configs(self):
        df = config_params_table(self.cfg)
        common = ", ".join(f"{k}={v}" for k, v in self.cfg.attrs["common"].items())
        self._write(df, "t01_lns_configs",
                    "LNS configurations (parameters that differ; all others shared)")
        with open(os.path.join(self.tab, "t01_lns_configs_common.txt"), "w") as f:
            f.write(common + "\n")

    def table_overall(self):
        rk = metrics.ranks(self.runs).mean()
        wn = metrics.wins(self.runs)
        for variant, methods in self.variants():
            rk_v = metrics.ranks(self.runs[self.runs.method.isin(methods)]).mean()
            rows = []
            for m in methods:
                r = self.runs[self.runs.method == m]
                row = {"Method": self.look[m]["label"], "Runs": len(r)}
                for T in self.time_points:
                    row[f"Gap @{T}s [%]"] = _gap_cell(r[f"gap_{T}"])
                row[f"Impr. over {self.start_name} [%]"] = mean_std(r.impr_final)
                row["Mean rank"] = f"{(rk_v if variant == 'reduced' else rk)[m]:.2f}"
                row["Best on # inst."] = int(wn.get(m, 0)) if variant == "full" else int(
                    metrics.wins(self.runs[self.runs.method.isin(methods)]).get(m, 0))
                rows.append(row)
            self._write(pd.DataFrame(rows), f"t02_overall_{variant}",
                        "Gap to the best known solution at each time point and final "
                        "improvement over the LNS start heuristic (mean ± std over runs, all "
                        "instances; runs without a solution yet are excluded and counted); "
                        "rank and wins use per-instance means (heuristics: one deterministic "
                        "run, available from their own runtime)")

    def table_by_size(self):
        for variant, methods in self.variants():
            rows = []
            for size in self.sizes:
                for m in methods:
                    r = self.runs[(self.runs.method == m) & (self.runs["size"] == size)]
                    if not len(r):
                        continue
                    row = {"Size": size, "Method": self.look[m]["label"]}
                    for T in self.time_points:
                        row[f"Gap @{T}s [%]"] = _gap_cell(r[f"gap_{T}"])
                    row[f"Impr. over {self.start_name} [%]"] = mean_std(r.impr_final)
                    row["Final makespan"] = mean_std(r.final, "{:.1f}")
                    rows.append(row)
            self._write(pd.DataFrame(rows), f"t03_by_size_{variant}",
                        "Gap to best known per size class (mean ± std over runs)")

    def table_per_instance(self):
        means = metrics.instance_means(self.runs)
        bks = self.runs.groupby("instance").bks.first()
        inst = self.instances.sort_values(["orders", "gen_seed"])
        for variant, methods in self.variants():
            rows = []
            for i in inst.itertuples():
                row = {"Instance": i.instance, "Size": i.size, "BKS": f"{bks[i.instance]:.0f}"}
                vals = means.loc[i.instance, methods]
                for m in methods:
                    v = vals[m]
                    mark = "*" if np.isclose(v, vals.min()) else ""
                    row[m] = f"{v:.1f}{mark}"
                rows.append(row)
            self._write(pd.DataFrame(rows), f"t04_per_instance_{variant}",
                        "Final makespan per instance (mean over runs; * = best mean)")

    def table_significance(self):
        for variant, methods in self.variants():
            others = [m for m in methods if m != self.best_lns]
            df = metrics.paired_tests(self.runs, self.best_lns, others)
            if df.empty:
                continue
            out = pd.DataFrame({
                "Method": [self.look[m]["label"] for m in df.method],
                "Instances": df.n,
                f"Median diff. vs {self.best_lns}": df.median_diff.map("{:+.1f}".format),
                f"{self.best_lns} better/tie/worse": [f"{a}/{b}/{c}" for a, b, c in
                                                      zip(df.ref_better, df.ties, df.ref_worse)],
                "p (Wilcoxon)": df.p.map("{:.4f}".format),
                "p (Holm)": df.p_holm.map("{:.4f}".format),
            })
            self._write(out, f"t05_significance_{variant}",
                        f"Best LNS config ({self.best_lns}) vs. other methods: paired Wilcoxon "
                        "signed-rank test on per-instance mean final makespan "
                        "(diff. > 0: the other method is worse)")

    def table_heuristics(self):
        rows = []
        for size in self.sizes:
            for m in self.heuristics:
                r = self.runs[(self.runs.method == m) & (self.runs["size"] == size)]
                rows.append({"Size": size, "Heuristic": self.look[m]["label"],
                             "Makespan": mean_std(r.final, "{:.1f}"),
                             "Gap to best known [%]": mean_std(r.gap_final),
                             f"Impr. over {self.start_name} [%]": mean_std(r.impr_final),
                             "Runtime [s]": mean_std(r.runtime, "{:.1f}")})
        self._write(pd.DataFrame(rows), "t06_heuristics",
                    "Constructive heuristics per size class (mean ± std over instances)")

    def table_lns_iterations(self):
        it = self.iters
        if not len(it):
            return
        lns_runs = self.runs[self.runs.family == "lns"]
        rows = []
        for size in self.sizes:
            for m in self.lns:
                gm = it[(it.method == m) & (it["size"] == size)]
                if not len(gm):
                    continue
                per_run = gm.groupby(["instance", "run_id"]).size()
                setup = lns_runs[(lns_runs.method == m) & (lns_runs["size"] == size)].setup_time
                for obj in self._phase_order(gm):
                    g = gm[gm.phase_obj == obj]
                    imp = g[g.phase_improving]
                    rows.append({
                        "Size": size, "Method": m, "Phase": obj_label(obj),
                        "Iterations / run": mean_std(
                            g.groupby(["instance", "run_id"]).size()
                            .reindex(per_run.index, fill_value=0), "{:.1f}"),
                        "Improve phase obj. [%]": f"{100 * g.phase_improving.mean():.1f}",
                        "New best makespan [%]": f"{100 * g.improving.mean():.1f}",
                        "Solutions / improving iter": mean_std(imp.num_discovered_solutions,
                                                               "{:.1f}"),
                        "Phase obj. impr. / improving iter [%]": mean_std(imp.phase_delta_pct,
                                                                          "{:.3f}"),
                        "Iteration time [s]": mean_std(g.total_iter_time),
                        "CP search share [%]":
                            f"{100 * g.solve_cpp_search.sum() / g.total_iter_time.sum():.1f}",
                        "Unfrozen vars [%]": mean_std(g.unfrozen_pct, "{:.1f}"),
                        "Setup [s]": mean_std(setup, "{:.1f}"),
                    })
        self._write(pd.DataFrame(rows), "t07_lns_iterations",
                    "LNS iteration statistics per configuration, size class and objective "
                    "phase. An iteration improves the phase objective if the CP finds a "
                    "solution better than the incumbent in the objective it optimises; "
                    "new best = it lowers the best makespan")

    @staticmethod
    def _phase_order(g):
        objs = list(pd.unique(g.phase_obj))
        return sorted(objs, key=lambda o: (o != "makespan", o))

    def _cycling(self):
        """LNS configs whose iterations optimise more than one objective."""
        n = self.iters.groupby("method").phase_obj.nunique()
        return [m for m in self.lns if n.get(m, 1) > 1]

    def table_strategies_makespan(self):
        """Strategies in makespan-phase iterations only."""
        it = self.iters
        if not len(it) or it.strategy_name.isna().all():
            return None
        rows = []
        for m in self.lns:
            g = it[(it.method == m) & (it.phase_obj == "makespan")]
            for name, s in g.groupby("strategy_name"):
                imp = s[s.phase_improving]
                rows.append({"Method": m, "Strategy": name, "Uses": len(s),
                             "Improves incumbent [%]": 100 * s.phase_improving.mean(),
                             "New best [%]": 100 * s.improving.mean(),
                             "Impr. / improving iter [%]": imp.phase_delta_pct.mean(),
                             "Unfrozen vars [%]": s.unfrozen_pct.mean()})
        df = pd.DataFrame(rows)
        self._write(_fmt(df, 1), "t08_strategies_makespan",
                    "Neighbourhood strategies in makespan-phase iterations, pooled over all "
                    "instances (improves incumbent: the CP found a solution with a lower "
                    "makespan than the incumbent; new best: the best makespan dropped)")
        return df

    def table_makespan_improvement_share(self):
        """Share of each config's total best-makespan improvement, by strategy and phase."""
        it = self.iters
        if not len(it) or it.strategy_name.isna().all():
            return None
        rows = []
        for m in self.lns:
            g = it[it.method == m]
            tot = g.delta.sum()
            if tot <= 0:
                continue
            for (obj, name), s in g.groupby(["phase_obj", "strategy_name"]):
                rows.append({"Method": m, "Phase": obj_label(obj), "Strategy": name,
                             "Column": f"{m} · {obj_label(obj)}" if m in self._cycling() else m,
                             "New-best iters": int(s.improving.sum()),
                             "Makespan improvement": s.delta.sum(),
                             "Share of total makespan improvement [%]":
                                 100 * s.delta.sum() / tot})
        df = pd.DataFrame(rows)
        self._write(_fmt(df.drop(columns="Column"), 1), "t09_makespan_improvement_share",
                    "Share of each configuration's total reduction of the best makespan, "
                    "by strategy and objective phase (sums to 100 per configuration)")
        return df

    def table_strategies_other_objectives(self):
        """Strategies in non-makespan phases of the objective-cycling configs."""
        it = self.iters
        cyc = self._cycling() if len(it) else []
        if not cyc:
            return None
        rows = []
        for m in cyc:
            g = it[(it.method == m) & (it.phase_obj != "makespan")]
            for obj, go in g.groupby("phase_obj"):
                tot = go.phase_delta.sum()
                for name, s in go.groupby("strategy_name"):
                    imp = s[s.phase_improving]
                    rows.append({
                        "Method": m, "Objective": obj_label(obj), "Strategy": name,
                        "Column": f"{m} · {obj_label(obj)}", "Uses": len(s),
                        "Improves objective [%]": 100 * s.phase_improving.mean(),
                        "Impr. / improving iter [%]": imp.phase_delta_pct.mean(),
                        "Share of total objective improvement [%]":
                            100 * s.phase_delta.sum() / tot if tot > 0 else np.nan,
                        "Accepted [%]": 100 * s.accepted.mean(),
                        "Makespan change / iter": s.makespan_change.mean(),
                        "New best makespan iters": int(s.improving.sum()),
                    })
        df = pd.DataFrame(rows)
        self._write(_fmt(df.drop(columns="Column"), 2), "t10_strategies_other_objectives",
                    "Neighbourhood strategies in the non-makespan phases of the configurations "
                    "with objective cycling. Improves objective: the CP found a solution better "
                    "than the incumbent in the phase objective; accepted: the incumbent changed "
                    "(acceptance is by makespan within the RRT threshold); makespan change: "
                    "mean change of the incumbent's makespan")
        return df

    def table_cp_startup(self):
        cp = self.runs[self.runs.family == "cp"]
        if not len(cp):
            return
        rows = [{"Size": s, "Extraction [s]": mean_std(g.extraction, "{:.1f}"),
                 "First incumbent [s]": mean_std(g.first_sol_time, "{:.1f}"),
                 "Warm-started runs": f"{int(g.warm_start.sum())}/{len(g)}"}
                for s in self.sizes for g in [cp[cp["size"] == s]]]
        self._write(pd.DataFrame(rows), "t11_cp_startup",
                    "Full-model CP start-up cost (mean ± std over runs). Warm-started runs hold "
                    "the start heuristic from t = 0, so this time counts as solution time")

    # ------------------------------------------------------------------ #
    # Solution structure
    # ------------------------------------------------------------------ #
    STRUCT_COLS = ["fetch_pick_ratio", "fetches_vs_start_pct", "refetch_pct",
                   "same_station_refetch_pct", "robot_work_vs_start_pct", "robot_util_pct",
                   "picker_util_pct", "workload_imb_pct", "makespan_imb_pct", "unused_stations"]
    STRUCT_LABELS = {"fetches_vs_start_pct": ("Fetches vs. start", "%"),
                     "robot_work_vs_start_pct": ("Robot work vs. start", "%")}

    def load_solutions(self):
        """Structure metrics of every final solution on disk (cached in data/)."""
        dirs = {os.path.basename(d): d for d in instance_dirs(self.root, self.pattern)}
        sol = solution_metrics(self.runs, dirs, os.path.join(self.dat, "solution_metrics.csv"))
        if not len(sol):
            self.sol = sol
            return sol
        sol["size"] = sol.instance.map(self.instances.set_index("instance")["size"])
        sol["gap_final"] = 100 * (sol.makespan - sol.instance.map(
            self.runs.groupby("instance").bks.first())) / sol.instance.map(
            self.runs.groupby("instance").bks.first())
        start = sol[sol.method == self.start_name].set_index("instance")
        sol["fetches_vs_start_pct"] = 100 * (sol.fetches / sol.instance.map(start.fetches) - 1)
        sol["robot_work_vs_start_pct"] = 100 * (
            sol.robot_work / sol.instance.map(start.robot_work) - 1)
        missing = self.runs[~self.runs.solution.isin(sol.solution)] \
            .groupby("method").size() if "solution" in self.runs else pd.Series(dtype=int)
        if len(missing):
            warnings.warn("runs without a solution file (left out of the structure "
                          f"metrics): {missing.to_dict()}")
        self.sol = sol
        return sol

    def _struct_label(self, col):
        label, unit = self.STRUCT_LABELS.get(col) or METRICS[col]
        return f"{label} [{unit}]" if unit else label

    def table_solution_structure(self):
        sol = self.sol
        if not len(sol):
            return
        cols = ["gap_final"] + self.STRUCT_COLS
        inst = sol.groupby(["method", "size", "instance"])[cols].mean().reset_index()
        for variant, methods in self.variants():
            rows = []
            for size in self.sizes:
                for m in methods:
                    g = inst[(inst["size"] == size) & (inst.method == m)]
                    if not len(g):
                        continue
                    row = {"Size": size, "Method": m, "Gap [%]": mean_std(g.gap_final)}
                    for c in self.STRUCT_COLS:
                        row[self._struct_label(c)] = mean_std(
                            g[c], "{:.3f}" if c == "fetch_pick_ratio" else "{:.1f}")
                    rows.append(row)
            self._write(pd.DataFrame(rows), f"t12_solution_structure_{variant}",
                        "Structure of the final solutions (mean ± std over the instances of a "
                        "size class; LNS and CP: mean over runs first). Start = "
                        f"{self.start_name}, the LNS start. Refetches: repeat visits of an "
                        "already fetched SKU; utilisation: busy share of the robot capacity "
                        "(MoveCap) and of the pickers over the makespan; imbalance: "
                        "(max - mean) / mean of the station pick workload and "
                        "(makespan - earliest station end) / makespan; unused stations: "
                        "stations without any pick")

    def figures_solution_structure(self):
        sol = self.sol
        if not len(sol):
            return
        labels = {**METRICS, **self.STRUCT_LABELS}
        cols = [c for c in self.STRUCT_COLS
                if c not in ("same_station_refetch_pct", "unused_stations")]
        for variant, methods in self.variants():
            plots.solution_structure(sol, methods, self.look, self.sizes, cols, labels,
                                     self.fig,
                                     f"fig11_solution_structure_{variant}",
                                     title="Structure of the final solutions")
            if self.start_name:
                plots.work_vs_utilisation(
                    sol, methods, self.look, self.sizes, self.start_name, self.fig,
                    f"fig12_robot_work_vs_utilisation_{variant}",
                    title="Makespan = robot work / (MoveCap · robot utilisation): "
                          f"change of both vs. {self.start_name}")
        if len(self.iters) and self.start_name:
            start = sol[sol.method == self.start_name].set_index("instance").fetches.to_dict()
            other = sol[sol.family != "lns"]
            refs = (other.groupby(["method", "family", "size", "instance"])
                    .fetches_vs_start_pct.mean().groupby(["method", "family", "size"]).mean()
                    .rename("value").reset_index())
            refs = refs[refs.method != self.start_name]
            plots.lns_fetches_over_time(self.iters, self.lns, self.look, self.sizes, start,
                                        refs, self.time_limit, self.fig,
                                        "fig13_lns_bin_fetches_over_time",
                                        title="Bin fetches of the LNS best solution over time "
                                              "(dashed: final CP and heuristic solutions)")

    # ------------------------------------------------------------------ #
    # Figures
    # ------------------------------------------------------------------ #
    def figures(self, strat_ms=None, share=None, other=None):
        r = self.runs
        for variant, methods in self.variants():
            plots.convergence(r, methods, self.look, self.sizes, self.time_limit, self.fig,
                              f"fig01_convergence_{variant}", bands=variant == "reduced",
                              highlight=[self.best_lns],
                              title="Gap to best known solution over time "
                                    f"(mean{' ± std' if variant == 'reduced' else ''} over runs)")
            plots.final_distribution(r, methods, self.look, self.sizes, self.fig,
                                     f"fig02_final_gap_{variant}",
                                     title=f"Final gap to best known after {self.time_limit} s "
                                           "(dots: runs; heuristics: one per instance)")
            plots.mean_ranks(r, methods, self.look, self.sizes, self.fig,
                             f"fig03_mean_rank_{variant}",
                             title="Mean rank of final makespan per instance")
            plots.instance_heatmap(r, methods, self.look, self.instances, self.fig,
                                   f"fig04_per_instance_{variant}",
                                   title="Mean final gap to best known per instance [%]")
        if self.heuristics:
            plots.heuristic_quality_time(r, self.heuristics, self.look, self.sizes, self.fig,
                                         "fig05_heuristics_quality_time",
                                         title="Constructive heuristics: quality vs. runtime")
        if len(self.iters):
            plots.lns_time_composition(self.iters, self.lns, self.look, self.sizes, self.fig,
                                       "fig06_lns_time_composition",
                                       title="Time composition of one LNS iteration")
        if strat_ms is not None and len(strat_ms):
            plots.strategy_heatmap(strat_ms, self.fig, "fig07_strategies_makespan_success",
                                   value="Improves incumbent [%]",
                                   title="Makespan phases: iterations that improve the "
                                         "incumbent's makespan [%]")
        if share is not None and len(share):
            plots.strategy_heatmap(share, self.fig, "fig08_makespan_improvement_share",
                                   value="Share of total makespan improvement [%]",
                                   column="Column", fmt="{:.1f}",
                                   title="Share of total best-makespan improvement per "
                                         "strategy [%] (each column sums to 100)")
        if other is not None and len(other):
            plots.strategy_heatmap(other, self.fig, "fig09_strategies_other_obj_success",
                                   value="Improves objective [%]", column="Column",
                                   title="Other-objective phases: iterations that improve the "
                                         "phase objective [%]")
            plots.strategy_heatmap(other, self.fig, "fig10_strategies_other_obj_share",
                                   value="Share of total objective improvement [%]",
                                   column="Column", fmt="{:.1f}",
                                   title="Other-objective phases: share of total improvement "
                                         "of the phase objective [%]")

    # ------------------------------------------------------------------ #
    def data(self):
        os.makedirs(self.dat, exist_ok=True)
        # config_key is the full config JSON; the method id identifies it (t01_lns_configs)
        drop = ["t", "v", "config_key"]
        self.runs.drop(columns=drop, errors="ignore").to_csv(
            os.path.join(self.dat, "runs.csv"), index=False)
        if len(self.iters):
            self.iters.drop(columns=drop, errors="ignore").to_csv(
                os.path.join(self.dat, "lns_iterations.csv.gz"), index=False)

    def readme(self):
        inst_gap = metrics.instance_means(self.runs, "gap_final").mean()
        counts = self.instances["size"].value_counts()
        per_size = ", ".join(f"{s}: {counts.get(s, 0)}" for s in self.sizes)
        lines = [f"# Method comparison: `{self.root}`", "",
                 f"{len(self.instances)} instances ({per_size}), "
                 f"{len(self.lns)} LNS configurations, CP, {len(self.heuristics)} heuristics. "
                 f"Time budget {self.time_limit} s.", "",
                 f"**Best LNS configuration:** {self.look[self.best_lns]['label']} "
                 f"(mean final gap to best known {inst_gap[self.best_lns]:.2f} %).", "",
                 f"Reduced variants show {', '.join(self.reduced)}.", "",
                 "## Mean final gap to best known (per-instance means)", "",
                 to_markdown(pd.DataFrame({
                     "Method": [self.look[m]["label"] for m in self.methods],
                     "Gap [%]": [f"{inst_gap[m]:.2f}" for m in self.methods]})),
                 "## Definitions", "",
                 "- Best known (BKS): best makespan any method found on the instance.",
                 "- Gap: (makespan - BKS) / BKS · 100.",
                 f"- Improvement: (start - makespan) / start · 100, start = {self.start_name} "
                 "(the LNS starting solution).",
                 "- Clocks: LNS and CP both start at t = 0 holding the start heuristic "
                 "(CP is warm-started from it); LNS model setup, CP model extraction and the "
                 "wait for CP's first incumbent all count as solution time. Heuristics are a "
                 "single point at their own runtime.",
                 "- Strategies are evaluated per objective phase: makespan phases in "
                 "t08/fig07, the share of the best-makespan improvement in t09/fig08, and the "
                 "other phases of objective-cycling configs in t10/fig09/fig10.", "",
                 "## Files", ""]
        for sub in ("tables", "figures"):
            lines.append(f"### {sub}")
            names = sorted({os.path.splitext(f)[0] for f in os.listdir(os.path.join(self.out, sub))})
            lines += [f"- {n}" for n in names] + [""]
        common = self.cfg.attrs["common"]
        lines += ["## Shared LNS parameters", "",
                  ", ".join(f"`{k}={v}`" for k, v in common.items()), ""]
        with open(os.path.join(self.out, "README.md"), "w") as f:
            f.write("\n".join(lines))

    def run(self):
        style.apply()
        os.makedirs(self.out, exist_ok=True)
        self.table_configs()
        self.table_overall()
        self.table_by_size()
        self.table_per_instance()
        self.table_significance()
        self.table_heuristics()
        self.table_lns_iterations()
        strat_ms = self.table_strategies_makespan()
        share = self.table_makespan_improvement_share()
        other = self.table_strategies_other_objectives()
        self.table_cp_startup()
        self.load_solutions()
        self.table_solution_structure()
        self.figures(strat_ms, share, other)
        self.figures_solution_structure()
        self.data()
        self.readme()
        return self

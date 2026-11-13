import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import data, metrics  # noqa: E402
from evaluation.configs import config_table  # noqa: E402
from evaluation.plots import _spread  # noqa: E402

BASE_CFG = {"runs": 2, "only_first_sol": False, "cp_objective_phase_schedule": "m",
            "cp_objective_phase_duration": 0, "time_limit": 100.0, "lns_model": "subproblem"}

CP_LOG = """ ! Initial process time : 1.50s (1.40s extraction + 0.10s propagation)
 *           100        1  2.00s              (gap is 100.0%)
 *            95       10  30.00s             (gap is 100.0%)
 ! Best objective         : 95 (gap is 100.0%)
"""


def _iter_rows(run_id, bests, discovered, objs=None, flows=None):
    rows = []
    objs = objs or ["makespan"] * len(bests)
    flows = flows or [1000.0] * len(bests)
    for i, (b, disc) in enumerate(zip(bests, discovered)):
        rows.append(dict(run_id=run_id, iteration=i, best=float(b), best_makespan=float(b),
                         current=float(b), current_makespan=float(b),
                         current_total_flow_time=float(flows[i]), cp_obj=objs[i],
                         total_iter_time=10.0, solve_time=9.0, lns_strategy_time=0.1,
                         constr_update_time=0.1, constr_generate_time=0.5,
                         solve_api_overhead=0.1, solve_cpp_presolve=1.0,
                         solve_cpp_search=8.0, merge_time=0.1, num_optimized=50,
                         strategy=0, statuses=6, discovered_solutions=disc,
                         num_discovered_solutions=len(disc), unfrozen_orders=set(),
                         unfrozen_skus=set()))
    return rows


def make_instance(root, name, start=100, cfgs=(), lns_bests=((98, 97, 97), (99, 99, 96))):
    d = os.path.join(root, name)
    os.makedirs(os.path.join(d, "experiments"))
    stations, orders = int(name.split("-")[1]), int(name.split("-")[4])
    json.dump({"stations": stations, "orders": orders, "gen_seed": 1,
               "heuristic": "load_balanced_greedy"}, open(os.path.join(d, "config.json"), "w"))
    with open(os.path.join(d, "heuristic_summary.txt"), "w") as f:
        f.write(f"LB Result | Feasible: True | Makespan: {start} | BinEvents: 1 | Time: 0.5s")
    with open(os.path.join(d, "heuristic_rdi_summary.txt"), "w") as f:
        f.write("rdi Result | Feasible: True | Makespan: 98 | BinEvents: 1 | Time: 20.0s\n"
                "Validation: PASSED")
    with open(os.path.join(d, "cplog_1.txt"), "w") as f:
        f.write(CP_LOG)
    for k, cfg in enumerate(cfgs):
        e = os.path.join(d, "experiments", f"0{k + 1}-11-2026_10-00-00")
        os.makedirs(e)
        json.dump(cfg, open(os.path.join(e, "experiment_config.json"), "w"))
        rows = []
        for rid, bests in enumerate(lns_bests, start=1):
            cycling = cfg.get("cp_objective_phase_schedule", "m") != "m"
            objs = ["makespan", "makespan", "total_flow_time"] if cycling else None
            disc = [[{"time": 2.0, "kpis": {"makespan": float(b), "total_flow_time": 990.0}}]
                    for b in bests]
            rows += _iter_rows(rid, bests, disc, objs=objs)
            with open(os.path.join(e, f"run_{rid}.log"), "w") as f:
                f.write("Setup time before first iteration: 5.0s\n"
                        "Optimized 000050/001000 (5.0)% | Strategy: random_orders (0) |\n")
        pd.DataFrame(rows).to_pickle(os.path.join(e, "experiment_dataframe.pkl"))
    return d


class TestTrajectories(unittest.TestCase):
    def test_values_at_running_min(self):
        t, v = data.make_traj([(5, 10), (1, 12), (8, 11)])
        np.testing.assert_array_equal(t, [1, 5, 8])
        np.testing.assert_array_equal(v, [12, 10, 10])
        out = data.values_at(t, v, [0, 1, 6, 100])
        self.assertTrue(np.isnan(out[0]))
        np.testing.assert_array_equal(out[1:], [12, 10, 10])

    def test_lns_trajectory_offsets(self):
        df = pd.DataFrame(_iter_rows(1, [100, 95], [[], [{"time": 3.0, "kpis": {"makespan": 95.0}}]]))
        (t, v), end = data.lns_trajectory(df, 100.0, setup_time=5.0)
        # second iteration starts at 5 + 10, pre-solve part 1 s, found at 3 s into the solve
        np.testing.assert_allclose(t, [0.0, 19.0])
        np.testing.assert_array_equal(v, [100, 95])
        self.assertAlmostEqual(end, 25.0)

    def test_phase_objective_columns(self):
        # iteration 1 optimises flow time: finds 950 < 1000 and is accepted with a
        # worse makespan (95 -> 96 is not a new best)
        disc = [[{"time": 1.0, "kpis": {"makespan": 95.0, "total_flow_time": 1000.0}}],
                [{"time": 1.0, "kpis": {"makespan": 96.0, "total_flow_time": 950.0}}]]
        df = pd.DataFrame(_iter_rows(1, [95, 95], disc, objs=["makespan", "total_flow_time"],
                                     flows=[1000.0, 950.0]))
        df.loc[1, ["current", "current_makespan"]] = 96.0
        cols = data._phase_objective_columns(df, 100.0)
        np.testing.assert_array_equal(cols["phase_prev"], [100.0, 1000.0])
        np.testing.assert_array_equal(cols["phase_found"], [95.0, 950.0])
        np.testing.assert_array_equal(cols["makespan_change"], [-5.0, 1.0])
        np.testing.assert_array_equal(cols["accepted"], [True, True])

    def test_cp_warm_start_from_zero(self):
        with tempfile.TemporaryDirectory() as root:
            d = make_instance(root, "GEN1-5-3-100-30-40")
            cold = data.load_cp(d, {"instance": "x"}, start_value=101.0)[0]
            warm = data.load_cp(d, {"instance": "x"}, start_value=100.0)[0]
        self.assertFalse(cold["warm_start"])
        np.testing.assert_array_equal(cold["t"], [2.0, 30.0])
        self.assertTrue(warm["warm_start"])
        np.testing.assert_array_equal(warm["t"], [0.0, 2.0, 30.0])
        self.assertEqual(warm["first_sol_time"], 2.0)

    def test_parse_cp_log(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(CP_LOG)
        inc, extraction, final = data.parse_cp_log(f.name)
        os.unlink(f.name)
        self.assertEqual(inc, [(2.0, 100.0), (30.0, 95.0)])
        self.assertEqual(extraction, 1.4)
        self.assertEqual(final, 95.0)


class TestSolutionMetrics(unittest.TestCase):
    def test_metrics_from_var_dict(self):
        from types import SimpleNamespace
        from evaluation.solutions import _from_var_dict, metrics
        inst = SimpleNamespace(S=[0, 1, 2], rt={7: 3, 8: 4}, rt_ret={7: 1, 8: 2}, movecap=2,
                               pickcap=1)
        on = lambda a, b: {"present": True, "start": a, "end": b}  # noqa: E731
        vs = {"B[0,7,0]": on(3, 6), "B[0,7,1]": on(8, 10), "B[1,7,0]": on(3, 5),
              "B[1,8,0]": on(4, 9), "Block[0,7,0]": on(0, 7), "B[2,8,1]": {"present": False},
              "P[0,0,7,0]": on(3, 5), "P[1,0,7,1]": on(8, 10), "P[2,0,7,0]": on(5, 6),
              "P[3,1,7,0]": on(3, 5), "P[4,1,8,0]": on(5, 8)}
        m = metrics(*_from_var_dict(vs), inst)
        self.assertEqual((m["makespan"], m["fetches"], m["lines"]), (10, 4, 5))
        self.assertAlmostEqual(m["fetch_pick_ratio"], 0.8)
        self.assertAlmostEqual(m["refetch_pct"], 50.0)            # 4 visits, 2 SKUs
        self.assertAlmostEqual(m["same_station_refetch_pct"], 25.0)  # (0, 7) twice
        self.assertAlmostEqual(m["robot_util_pct"], 100 * (3 * 4 + 6) / (2 * 10))
        self.assertAlmostEqual(m["picker_util_pct"], 100 * 10 / (3 * 10))
        self.assertEqual(m["unused_stations"], 1)
        self.assertAlmostEqual(m["makespan_imb_pct"], 100.0)
        self.assertAlmostEqual(m["workload_imb_pct"], 100 * (5 - 10 / 3) / (10 / 3))


class TestConfigs(unittest.TestCase):
    def test_missing_key_gets_old_default(self):
        a = dict(BASE_CFG)  # no combine_strat_with_tail -> "all"
        b = dict(BASE_CFG, combine_strat_with_tail="all")
        self.assertEqual(data.config_key(a), data.config_key(b))

    def test_labels_name_only_differing_params(self):
        exps = pd.DataFrame([
            dict(instance="x", date=pd.Timestamp("2026-11-01"),
                 config_key=data.config_key(dict(BASE_CFG, only_first_sol=True))),
            dict(instance="x", date=pd.Timestamp("2026-11-02"),
                 config_key=data.config_key(dict(BASE_CFG, cp_objective_phase_schedule="mmmf",
                                                 cp_objective_phase_duration=4))),
        ])
        t = config_table(exps)
        self.assertEqual(list(t.method), ["LNS-1", "LNS-2"])
        self.assertEqual(list(t.label), ["first-sol · obj=m", "all-sols · obj=mmmf/4"])


class TestMetrics(unittest.TestCase):
    def test_holm(self):
        np.testing.assert_allclose(metrics.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_spread_keeps_order_and_gap(self):
        ys = np.array([5.0, 5.05, 0.0, 0.1, 0.2])
        out = _spread(ys, 0.5)
        self.assertTrue(np.all(np.diff(out[np.argsort(ys)]) >= 0.5 - 1e-9))
        self.assertAlmostEqual(out[[0, 1]].mean(), ys[[0, 1]].mean())


class TestEndToEnd(unittest.TestCase):
    def test_compare_report(self):
        from evaluation.reports.compare import Comparison
        cfgs = (dict(BASE_CFG, only_first_sol=True),
                dict(BASE_CFG, cp_objective_phase_schedule="mmmf", cp_objective_phase_duration=4))
        with tempfile.TemporaryDirectory() as root:
            for name in ("GEN1-5-3-100-30-40", "GEN2-5-3-100-30-40", "GEN1-10-3-100-60-40"):
                make_instance(root, name, cfgs=cfgs)
            out = os.path.join(root, "out")
            c = Comparison(root, out, time_points=(10, 100)).run()
            self.assertEqual(c.lns, ["LNS-1", "LNS-2"])
            self.assertEqual(c.methods[-1], "LB-SGC")  # worst heuristic last
            self.assertEqual(c.reduced[1:], ["CP", "RDI"])
            # BKS = 95 (CP); LNS ends at 96 or 97 -> mean final 96.5
            lns = c.runs[c.runs.method == "LNS-1"]
            self.assertTrue((lns.bks == 95).all())
            self.assertAlmostEqual(lns.final.mean(), 96.5)
            for f in ("tables/t02_overall_full.csv", "tables/t02_overall_reduced.tex",
                      "tables/t08_strategies_makespan.csv",
                      "tables/t09_makespan_improvement_share.csv",
                      "figures/fig01_convergence_reduced.pdf",
                      "figures/fig08_makespan_improvement_share.pdf",
                      "tables/t10_strategies_other_objectives.csv",
                      "figures/fig09_strategies_other_obj_success.pdf",
                      "data/runs.csv", "README.md"):
                self.assertTrue(os.path.isfile(os.path.join(out, f)), f)


if __name__ == "__main__":
    unittest.main()

#include <ilcp/cp.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <iostream>
#include <optional>

namespace py = pybind11;

class CpLnsModel {
private:
    IloEnv env;
    IloModel model;
    IloCP cp;
    std::unordered_map<std::string, IloIntervalVar> vars;
    std::unordered_map<std::string, std::vector<IloConstraint>> active_freeze_constraints;
    IloObjective makespan_obj;

    std::string make_key(const std::string& prefix, int o, int s) { return prefix + "[" + std::to_string(o) + "," + std::to_string(s) + "]"; }
    std::string make_key(const std::string& prefix, int o, int s, int ln) { return prefix + "[" + std::to_string(o) + "," + std::to_string(s) + "," + std::to_string(ln) + "]"; }
    std::string make_key_s_k_e(const std::string& prefix, int s, int k, int e) { return prefix + "[" + std::to_string(s) + "," + std::to_string(k) + "," + std::to_string(e) + "]"; }
    std::string make_key_o_s_k_e(const std::string& prefix, int o, int s, int k, int e) { return prefix + "[" + std::to_string(o) + "," + std::to_string(s) + "," + std::to_string(k) + "," + std::to_string(e) + "]"; }

public:
    CpLnsModel(const std::vector<int>& S,
               const std::vector<int>& L,
               const std::vector<int>& K,
               const std::map<int, std::vector<int>>& orders_req,
               const std::map<int, int>& rt,
               const std::map<int, int>& p,
               const std::map<int, int>& N,
               std::optional<int> move_cap,
               const std::map<int, int>& rt_return,
               const std::vector<int>& active_K,
               const std::map<int, int>& U,
               int horizon,
               bool add_symmetry_breaking) {
        
        model = IloModel(env);
        cp = IloCP(model);
        cp.setOut(env.getNullStream());

        std::vector<int> O;
        for (const auto& pair : orders_req) {
            O.push_back(pair.first);
        }
        
        for (int o : O) {
            for (int s : S) {
                std::string name_I_os = make_key("I_os", o, s);
                IloIntervalVar iv(env);
                iv.setOptional();
                iv.setName(name_I_os.c_str());
                vars[name_I_os] = iv;
                
                for (int ln : L) {
                    std::string name_lane = make_key("I_os_lane", o, s, ln);
                    IloIntervalVar iv_lane(env);
                    iv_lane.setOptional();
                    iv_lane.setName(name_lane.c_str());
                    vars[name_lane] = iv_lane;
                }
            }
        }

        for (int o : O) {
            for (int k : orders_req.at(o)) {
                for (int s : S) {
                    std::string name_C = "C[" + std::to_string(o) + "," + std::to_string(k) + "," + std::to_string(s) + "]";
                    IloIntervalVar iv(env, p.at(k));
                    iv.setOptional();
                    iv.setName(name_C.c_str());
                    vars[name_C] = iv;
                }
            }
        }

        for (int s : S) {
            for (int k : active_K) {
                for (int e = 0; e < U.at(k); ++e) {
                    std::vector<IloIntervalVar> all_picks_for_this_copy;
                    for (int o : O) {
                        const auto& reqs = orders_req.at(o);
                        if (std::find(reqs.begin(), reqs.end(), k) != reqs.end()) {
                            std::string name_P = make_key_o_s_k_e("P", o, s, k, e);
                            IloIntervalVar iv_P(env, p.at(k));
                            iv_P.setOptional();
                            iv_P.setName(name_P.c_str());
                            vars[name_P] = iv_P;
                            all_picks_for_this_copy.push_back(iv_P);
                        }
                    }
                    
                    std::string name_F = make_key_s_k_e("F", s, k, e);
                    IloIntervalVar iv_F(env, rt.at(k));
                    iv_F.setOptional();
                    iv_F.setName(name_F.c_str());
                    vars[name_F] = iv_F;
                    
                    std::string name_R = make_key_s_k_e("R", s, k, e);
                    int rt_ret_val = (rt_return.find(k) != rt_return.end()) ? rt_return.at(k) : rt.at(k);
                    IloIntervalVar iv_R(env, rt_ret_val);
                    iv_R.setOptional();
                    iv_R.setName(name_R.c_str());
                    vars[name_R] = iv_R;
                    
                    std::string name_B = make_key_s_k_e("B", s, k, e);
                    IloIntervalVar iv_B(env);
                    iv_B.setOptional();
                    iv_B.setName(name_B.c_str());
                    vars[name_B] = iv_B;
                    
                    model.add(IloPresenceOf(env, iv_B) == IloPresenceOf(env, iv_F));
                    model.add(IloPresenceOf(env, iv_R) == IloPresenceOf(env, iv_B));
                    
                    if (!all_picks_for_this_copy.empty()) {
                        IloIntExprArray presences(env, all_picks_for_this_copy.size());
                        for (size_t i = 0; i < all_picks_for_this_copy.size(); ++i) {
                            presences[i] = IloPresenceOf(env, all_picks_for_this_copy[i]);
                        }
                        model.add(IloPresenceOf(env, iv_B) == IloMax(presences));
                    } else {
                        model.add(IloPresenceOf(env, iv_B) == 0);
                    }
                    
                    for (int o : O) {
                        const auto& reqs = orders_req.at(o);
                        if (std::find(reqs.begin(), reqs.end(), k) != reqs.end()) {
                            IloIntervalVar iv_P = vars[make_key_o_s_k_e("P", o, s, k, e)];
                            model.add(IloEndBeforeStart(env, iv_F, iv_P));
                            model.add(IloEndBeforeStart(env, iv_P, iv_R));
                        }
                    }
                    
                    model.add(IloStartAtEnd(env, iv_B, iv_F));
                    model.add(IloEndAtStart(env, iv_B, iv_R));
                    
                    std::string name_Block = make_key_s_k_e("Block", s, k, e);
                    IloIntervalVar iv_Block(env);
                    iv_Block.setOptional();
                    iv_Block.setName(name_Block.c_str());
                    vars[name_Block] = iv_Block;
                    
                    IloIntervalVarArray span_arr(env, 2);
                    span_arr[0] = iv_F;
                    span_arr[1] = iv_R;
                    model.add(IloSpan(env, iv_Block, span_arr));
                }
            }
        }
        
        if (horizon == 0) {
            horizon = 0;
            for (int k : active_K) {
                int rt_ret_val = (rt_return.find(k) != rt_return.end()) ? rt_return.at(k) : rt.at(k);
                horizon += (rt.at(k) + p.at(k) + rt_ret_val) * U.at(k);
            }
        }
        
        if (move_cap.has_value()) {
            IloCumulFunctionExpr moves(env);
            for (int s : S) {
                for (int k : active_K) {
                    for (int e = 0; e < U.at(k); ++e) {
                        IloIntervalVar iv_F = vars[make_key_s_k_e("F", s, k, e)];
                        IloIntervalVar iv_R = vars[make_key_s_k_e("R", s, k, e)];
                        moves += IloPulse(iv_F, 1);
                        moves += IloPulse(iv_R, 1);
                    }
                }
            }
            model.add(IloAlwaysIn(env, moves, 0, horizon, 0, move_cap.value()));
        }

        for (int o : O) {
            IloIntExpr sum_s(env);
            for (int s : S) {
                sum_s += IloPresenceOf(env, vars[make_key("I_os", o, s)]);
            }
            model.add(sum_s == 1);
        }
        
        for (int o : O) {
            for (int s : S) {
                IloIntervalVar iv_I_os = vars[make_key("I_os", o, s)];
                IloIntervalVarArray alts(env, L.size());
                for (size_t i = 0; i < L.size(); ++i) {
                    alts[i] = vars[make_key("I_os_lane", o, s, L[i])];
                }
                model.add(IloAlternative(env, iv_I_os, alts));
            }
        }
        
        for (int s : S) {
            for (int ln : L) {
                IloIntervalVarArray lane_set(env);
                for (int o : O) {
                    lane_set.add(vars[make_key("I_os_lane", o, s, ln)]);
                }
                if (lane_set.getSize() >= 2) {
                    model.add(IloNoOverlap(env, lane_set));
                }
            }
        }

        for (int o : O) {
            const auto& R_o = orders_req.at(o);
            for (int s : S) {
                IloIntervalVar iv_I_os = vars[make_key("I_os", o, s)];
                if (!R_o.empty()) {
                    IloIntervalVarArray c_arr(env, R_o.size());
                    for (size_t i = 0; i < R_o.size(); ++i) {
                        c_arr[i] = vars["C[" + std::to_string(o) + "," + std::to_string(R_o[i]) + "," + std::to_string(s) + "]"];
                    }
                    model.add(IloSpan(env, iv_I_os, c_arr));
                } else {
                    model.add(IloLengthOf(iv_I_os) == 0);
                }
                
                for (int k : R_o) {
                    IloIntervalVar iv_C = vars["C[" + std::to_string(o) + "," + std::to_string(k) + "," + std::to_string(s) + "]"];
                    model.add(IloPresenceOf(env, iv_C) == IloPresenceOf(env, iv_I_os));
                }
            }
        }

        for (int o : O) {
            for (int k : orders_req.at(o)) {
                for (int s : S) {
                    IloIntervalVar iv_C = vars["C[" + std::to_string(o) + "," + std::to_string(k) + "," + std::to_string(s) + "]"];
                    int Uk = U.at(k);
                    if (Uk <= 0) {
                        model.add(IloPresenceOf(env, iv_C) == 0);
                    } else {
                        IloIntervalVarArray candidates(env, Uk);
                        for (int e = 0; e < Uk; ++e) {
                            candidates[e] = vars[make_key_o_s_k_e("P", o, s, k, e)];
                        }
                        model.add(IloAlternative(env, iv_C, candidates));
                    }
                }
            }
        }
        
        for (int s : S) {
            IloIntervalVarArray bins_here(env);
            for (int k : active_K) {
                for (int e = 0; e < U.at(k); ++e) {
                    bins_here.add(vars[make_key_s_k_e("B", s, k, e)]);
                }
            }
            if (bins_here.getSize() >= 2) {
                model.add(IloNoOverlap(env, bins_here));
            }
        }
        
        for (int k : active_K) {
            IloIntervalVarArray family(env);
            for (int s : S) {
                for (int e = 0; e < U.at(k); ++e) {
                    family.add(vars[make_key_s_k_e("Block", s, k, e)]);
                }
            }
            if (family.getSize() <= 1) continue;
            
            if (N.at(k) <= 1) {
                if (family.getSize() >= 2) {
                    model.add(IloNoOverlap(env, family));
                }
            }
            
            if (N.at(k) > family.getSize()) continue;
            
            if (move_cap.has_value() && (size_t)N.at(k) >= (S.size() + (size_t)move_cap.value())) continue;
            
            IloCumulFunctionExpr bin_usage(env);
            for (int s : S) {
                for (int e = 0; e < U.at(k); ++e) {
                    bin_usage += IloPulse(vars[make_key_s_k_e("Block", s, k, e)], 1);
                }
            }
            model.add(IloAlwaysIn(env, bin_usage, 0, horizon, 0, N.at(k)));
        }
        
        if (add_symmetry_breaking) {
            for (int s : S) {
                for (size_t i = 0; i < L.size() - 1; ++i) {
                    IloIntExpr sum1(env), sum2(env);
                    for (int o : O) {
                        sum1 += IloPresenceOf(env, vars[make_key("I_os_lane", o, s, L[i])]);
                        sum2 += IloPresenceOf(env, vars[make_key("I_os_lane", o, s, L[i+1])]);
                    }
                    model.add(sum1 >= sum2);
                }
            }
            
            for (int s : S) {
                for (int k : active_K) {
                    int Uk = U.at(k);
                    for (int e = 0; e < Uk - 1; ++e) {
                        IloIntervalVar b_e1 = vars[make_key_s_k_e("B", s, k, e+1)];
                        IloIntervalVar b_e = vars[make_key_s_k_e("B", s, k, e)];
                        model.add(IloIfThen(env, IloPresenceOf(env, b_e1) == 1, IloPresenceOf(env, b_e) == 1));
                        model.add(IloEndBeforeStart(env, b_e, b_e1));
                    }
                }
            }
        }
        
        if (horizon > 0) {
            for (int o : O) {
                for (int s : S) {
                    model.add(IloEndOf(vars[make_key("I_os", o, s)]) <= horizon);
                }
            }
        }
        
        IloIntExprArray per_order_ends(env, O.size());
        for (size_t i = 0; i < O.size(); ++i) {
            int o = O[i];
            IloIntExprArray ends(env, S.size());
            for (size_t j = 0; j < S.size(); ++j) {
                int s = S[j];
                ends[j] = IloEndOf(vars[make_key("I_os", o, s)]);
            }
            per_order_ends[i] = IloMax(ends);
        }
        
        IloIntExpr makespan_expr = IloMax(per_order_ends);
        makespan_obj = IloMinimize(env, makespan_expr);
        model.add(makespan_obj);
    }
    
    ~CpLnsModel() {
        env.end();
    }
    
    void apply_warm_start(const std::vector<std::string>& names, const std::vector<int>& present, const std::vector<int>& start, const std::vector<int>& end) {
        IloSolution sp(env);
        for (size_t i = 0; i < names.size(); ++i) {
            const std::string& name = names[i];
            if (vars.find(name) != vars.end()) {
                IloIntervalVar iv = vars[name];
                if (present[i]) {
                    sp.setPresent(iv);
                    sp.setStart(iv, start[i]);
                    sp.setEnd(iv, end[i]);
                    sp.setSize(iv, end[i] - start[i]);
                } else {
                    sp.setAbsent(iv);
                }
            }
        }
        cp.setStartingPoint(sp);
    }
    
    void apply_delta_freezing(const std::vector<std::string>& to_freeze, const std::vector<int>& present, const std::vector<int>& start, const std::vector<int>& end, const std::vector<std::string>& to_unfreeze) {
        IloExtractableArray bulk_remove(env);
        for (const std::string& name : to_unfreeze) {
            auto it = active_freeze_constraints.find(name);
            if (it != active_freeze_constraints.end()) {
                for (IloConstraint c : it->second) {
                    bulk_remove.add(c);
                }
                active_freeze_constraints.erase(it);
            }
        }
        if (bulk_remove.getSize() > 0) {
            model.remove(bulk_remove);
        }
        bulk_remove.end();
        
        IloExtractableArray bulk_add(env);
        for (size_t i = 0; i < to_freeze.size(); ++i) {
            const std::string& name = to_freeze[i];
            if (active_freeze_constraints.find(name) == active_freeze_constraints.end()) {
                std::vector<IloConstraint> constraints;
                if (vars.find(name) != vars.end()) {
                    IloIntervalVar iv = vars[name];
                    if (present[i]) {
                        IloConstraint p_c = (IloPresenceOf(env, iv) == 1);
                        IloConstraint s_c = (IloStartOf(iv) == start[i]);
                        IloConstraint e_c = (IloEndOf(iv) == end[i]);
                        bulk_add.add(p_c);
                        bulk_add.add(s_c);
                        bulk_add.add(e_c);
                        constraints.push_back(p_c);
                        constraints.push_back(s_c);
                        constraints.push_back(e_c);
                    } else {
                        IloConstraint p_c = (IloPresenceOf(env, iv) == 0);
                        bulk_add.add(p_c);
                        constraints.push_back(p_c);
                    }
                }
                active_freeze_constraints[name] = constraints;
            }
        }
        if (bulk_add.getSize() > 0) {
            model.add(bulk_add);
        }
        bulk_add.end();
    }
    
    py::dict solve(double time_limit, int improvement_makespan = -1, int workers = 1, std::string presolve = "Auto", std::string search_type = "Auto") {
        cp.setParameter(IloCP::TimeLimit, time_limit);
        cp.setParameter(IloCP::Workers, workers);
        cp.setParameter(IloCP::LogVerbosity, IloCP::Quiet);

        if (presolve == "Off") {
            cp.setParameter(IloCP::Presolve, IloCP::Off);
        } else if (presolve == "On") {
            cp.setParameter(IloCP::Presolve, IloCP::On);
        }

        if (search_type == "DepthFirst") {
            cp.setParameter(IloCP::SearchType, IloCP::DepthFirst);
        } else if (search_type == "Restart") {
            cp.setParameter(IloCP::SearchType, IloCP::Restart);
        } else if (search_type == "MultiPoint") {
            cp.setParameter(IloCP::SearchType, IloCP::MultiPoint);
        } else if (search_type == "Auto") {
            cp.setParameter(IloCP::SearchType, IloCP::Auto);
        }

        IloConstraint impr_constr;
        if (improvement_makespan != -1) {
            impr_constr = (makespan_obj.getExpr() <= improvement_makespan);
            model.add(impr_constr);
        }
        
        bool found = cp.solve();
        
        py::dict result;
        if (found) {
            result["status"] = (cp.getInfo(IloCP::FailStatus) == IloCP::SearchStopped) ? "Feasible" : "Optimal";
            result["objective"] = cp.getObjValue();
            
            double total_time = cp.getInfo(IloCP::SolveTime);
            double extraction_time = cp.getInfo(IloCP::ExtractionTime);
            
            result["TotalTime"] = total_time;
            result["ExtractionTime"] = extraction_time;
            result["SolveTime"] = total_time - extraction_time;
            
            py::dict var_sols;
            for (const auto& pair : vars) {
                const std::string& name = pair.first;
                IloIntervalVar iv = pair.second;
                if (cp.isPresent(iv)) {
                    py::dict iv_sol;
                    iv_sol["present"] = true;
                    iv_sol["start"] = cp.getStart(iv);
                    iv_sol["end"] = cp.getEnd(iv);
                    iv_sol["size"] = cp.getSize(iv);
                    var_sols[name.c_str()] = iv_sol;
                } else {
                    py::dict iv_sol;
                    iv_sol["present"] = false;
                    var_sols[name.c_str()] = iv_sol;
                }
            }
            result["var_solutions"] = var_sols;
        } else {
            result["status"] = "Unknown";
            result["objective"] = py::none();
            result["var_solutions"] = py::dict();
            
            double total_time = cp.getInfo(IloCP::SolveTime);
            double extraction_time = cp.getInfo(IloCP::ExtractionTime);
            
            result["TotalTime"] = total_time;
            result["ExtractionTime"] = extraction_time;
            result["SolveTime"] = total_time - extraction_time;
        }
        
        if (improvement_makespan != -1) {
            model.remove(impr_constr);
        }
        
        return result;
    }
};

PYBIND11_MODULE(cp_lns_core, m) {
    py::class_<CpLnsModel>(m, "CpLnsModel")
        .def(py::init<std::vector<int>, std::vector<int>, std::vector<int>, std::map<int, std::vector<int>>, std::map<int, int>, std::map<int, int>, std::map<int, int>, std::optional<int>, std::map<int, int>, std::vector<int>, std::map<int, int>, int, bool>())
        .def("apply_warm_start", &CpLnsModel::apply_warm_start)
        .def("apply_delta_freezing", &CpLnsModel::apply_delta_freezing)
        .def("solve", &CpLnsModel::solve, py::arg("time_limit"), py::arg("improvement_makespan") = -1, py::arg("workers") = 1, py::arg("presolve") = "Auto", py::arg("search_type") = "Auto");
}

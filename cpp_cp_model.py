from collections import defaultdict

import cp_lns_core

from cp_model_utils import MockIntervalVar


def build_model(instance, add_symmetry_breaking: bool, horizon: int):
    S, L, K, orders_req, rt, p, N = instance
    move_cap = instance.movecap
    rt_return = instance.rt_ret

    O = sorted(orders_req.keys())

    orders_demanding_sku = defaultdict(set)
    need_count = defaultdict(int)
    for o in O:
        for k in orders_req[o]:
            orders_demanding_sku[k].add(o)
            need_count[k] += 1
    active_K = [k for k in K if need_count[k] > 0]

    U = {k: need_count[k] for k in K}

    # Initialize the Pybind11 C++ model
    cpp_model = cp_lns_core.CpLnsModel(
        S,
        L,
        K,
        orders_req,
        rt,
        p,
        N,
        move_cap,
        rt_return,
        active_K,
        U,
        horizon,
        add_symmetry_breaking,
    )

    # Reconstruct handles dictionary with MockIntervalVar
    I_os, I_os_lane = {}, {}
    for o in O:
        for s in S:
            I_os[(o, s)] = MockIntervalVar(f"I_os[{o},{s}]")
            for ln in L:
                I_os_lane[(o, s, ln)] = MockIntervalVar(f"I_os_lane[{o},{s},{ln}]")

    C = {}
    for o in O:
        for k in orders_req[o]:
            for s in S:
                C[(o, k, s)] = MockIntervalVar(f"C[{o},{k},{s}]")

    P, F, R, B, Block = {}, {}, {}, {}, {}
    for s in S:
        for k in active_K:
            for e in range(U[k]):
                for o in O:
                    if k in orders_req[o]:
                        P[(o, s, k, e)] = MockIntervalVar(f"P[{o},{s},{k},{e}]")

                F[(s, k, e)] = MockIntervalVar(f"F[{s},{k},{e}]")
                R[(s, k, e)] = MockIntervalVar(f"R[{s},{k},{e}]")
                B[(s, k, e)] = MockIntervalVar(f"B[{s},{k},{e}]")
                Block[(s, k, e)] = MockIntervalVar(f"Block[{s},{k},{e}]")

    all_intervals_flat = {}
    for dic in [I_os, I_os_lane, C, P, F, R, B, Block]:
        for val in dic.values():
            all_intervals_flat[val.get_name()] = val

    handles = {
        "I_os_lane": I_os_lane,
        "I_os": I_os,
        "C": C,
        "P": P,
        "F": F,
        "R": R,
        "B": B,
        "Block": Block,
        "U": U,
        "orders_req": orders_req,
        "rt": rt,
        "rt_return": rt_return,
        "p": p,
        "S": S,
        "L": L,
        "K": K,
        "active_K": active_K,
        "orders_demanding_sku": orders_demanding_sku,
        "O": O,
        "N": N,
        "move_cap": move_cap,
        "all_intervals_flat": all_intervals_flat,
    }

    return cpp_model, handles


def export_model(model, filename: str):
    """Exports the C++ CP model to a .cpo file."""
    model.export_model(filename)


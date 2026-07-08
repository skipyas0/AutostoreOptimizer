from instance import Instance
from datagen import generate_data

def test_instance_properties():
    # Test creation and properties/aliases
    S = [0, 1]
    L = [0, 1, 2]
    K = [0, 1, 2, 3]
    orders_req = {0: [0, 1], 1: [2, 3]}
    rt = {0: 10, 1: 20, 2: 30, 3: 40}
    p = {0: 4, 1: 4, 2: 4, 3: 4}
    N = {0: 1, 1: 1, 2: 2, 3: 2}

    inst = Instance(S, L, K, orders_req, rt, p, N)

    assert inst.S == S
    assert inst.L == L
    assert inst.K == K
    assert inst.orders_requirements == orders_req
    assert inst.orders_req == orders_req
    assert inst.rt == rt
    assert inst.p == p
    assert inst.N == N

    # Test O and rt_ret properties
    assert inst.O == [0, 1]
    assert inst.rt_ret == rt

    # Test unpacking
    S_u, L_u, K_u, req_u, rt_u, p_u, N_u = inst
    assert S_u == S
    assert L_u == L
    assert K_u == K
    assert req_u == orders_req
    assert rt_u == rt
    assert p_u == p
    assert N_u == N

def test_instance_statistics(capsys):
    S = [0]
    L = [0, 1]
    K = [0, 1, 2]
    orders_req = {0: [0, 1], 1: [1, 2]}
    rt = {0: 10, 1: 20, 2: 30}
    p = {0: 4, 1: 4, 2: 4}
    N = {0: 1, 1: 2, 2: 3}

    inst = Instance(S, L, K, orders_req, rt, p, N)
    stats = inst.get_statistics()

    assert stats["num_stations"] == 1
    assert stats["num_lanes"] == 2
    assert stats["num_orders"] == 2
    assert stats["num_skus"] == 3
    assert stats["used_skus"] == 3
    assert stats["unused_skus"] == 0
    assert stats["order_size_min"] == 2
    assert stats["order_size_max"] == 2
    assert stats["rt_mean"] == 20.0
    assert stats["p_mean"] == 4.0
    assert stats["n_mean"] == 2.0
    assert stats["total_pick_lines"] == 4

    # Test print_summary
    inst.print_summary()
    captured = capsys.readouterr()
    assert "AutoStore Instance Summary" in captured.out
    assert "Stations:  1" in captured.out
    assert "Lanes/st:  2" in captured.out
    assert "Orders:    2" in captured.out


def test_instance_custom_rt_ret():
    S = [0]
    L = [0, 1]
    K = [0, 1]
    orders_req = {0: [0, 1]}
    rt = {0: 10, 1: 20}
    p = {0: 4, 1: 4}
    N = {0: 1, 1: 1}
    rt_ret = {0: 15, 1: 25}

    inst = Instance(S, L, K, orders_req, rt, p, N, rt_ret=rt_ret)
    assert inst.rt_ret == rt_ret

    # Test that modifying via setter works
    new_rt_ret = {0: 12, 1: 22}
    inst.rt_ret = new_rt_ret
    assert inst.rt_ret == new_rt_ret

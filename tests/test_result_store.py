import os
import pickle
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import result_store as rs  # noqa: E402
from scripts.compact_results import same  # noqa: E402


@pytest.fixture
def tmp():
    with tempfile.TemporaryDirectory() as d:
        yield d


def test_barcode_roundtrip(tmp):
    rng = np.random.default_rng(0)
    m = np.full((37, 1001), -1, dtype=np.int8)
    m[rng.random(m.shape) < 0.03] = 5
    m[3, :10] = 0
    p = os.path.join(tmp, "b.npz")
    rs.save_barcode(p, m)
    back = rs.load_barcode(p)
    assert back.dtype == m.dtype and np.array_equal(back, m)
    np.save(os.path.join(tmp, "old.npy"), m)
    assert np.array_equal(rs.load_barcode(os.path.join(tmp, "old.npy")), m)


def test_barcode_empty(tmp):
    p = os.path.join(tmp, "b.npz")
    rs.save_barcode(p, np.full((0, 4), -1, dtype=np.int8))
    assert rs.load_barcode(p).shape == (0, 4)


def test_compact_pickle_exact(tmp):
    init = {f"P[{o},{s},{k}]": ({"present": True, "start": o, "end": o + 3} if k % 3 == 0
                                else {"present": False, "start": None, "end": None})
            for o in range(20) for s in range(3) for k in range(5)}
    init["I_os[1,2]"] = {"present": False}
    init["odd name"] = {"present": True, "start": 1, "end": 2}
    init["Q[-0,07]"] = {"present": True, "start": 1, "end": 2}  # not canonical ints
    init["F[1,2]"] = {"x": 1.5}  # float -> stored verbatim
    init["X[1]"] = 7  # non-dict value
    deltas = [{}, {"P[0,0,0]": {"present": True, "start": 9, "end": 12}},
              {"NEW[3]": {"present": True, "start": 2**40, "end": -5}}]
    payload = [{"run_id": 1, "initial_solution": init, "deltas": deltas},
               {"run_id": 2, "initial_solution": dict(reversed(init.items())), "deltas": []}]
    p = os.path.join(tmp, "s.npz")
    rs.dump(payload, p)
    assert same(rs.load(p), payload)


def test_shared_references_preserved(tmp):
    vd = {"B[1,2]": {"present": True, "start": 0, "end": 4}}
    obj = {"a": vd, "b": [vd, {"k": vd}]}
    p = os.path.join(tmp, "s.npz")
    rs.dump(obj, p)
    back = rs.load(p)
    assert back["a"] is back["b"][0] is back["b"][1]["k"]
    assert same(back, obj)


def test_load_legacy_and_find(tmp):
    obj = {"P[1]": {"present": False}}
    with open(os.path.join(tmp, "x.pkl"), "wb") as f:
        pickle.dump(obj, f)
    assert rs.find(os.path.join(tmp, "x")) == os.path.join(tmp, "x.pkl")
    assert rs.load(os.path.join(tmp, "x.pkl")) == obj
    rs.dump(obj, os.path.join(tmp, "x.npz"))
    assert rs.find(os.path.join(tmp, "x.pkl")) == os.path.join(tmp, "x.npz")
    assert rs.find_all(os.path.join(tmp, "*")) == [os.path.join(tmp, "x.npz")]
    assert rs.find(os.path.join(tmp, "missing")) is None

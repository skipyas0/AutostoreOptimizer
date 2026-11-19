"""Compact on-disk storage of LNS / CP result data.

Two file kinds, both ordinary compressed ``.npz`` archives:

Barcodes (``save_barcode`` / ``load_barcode``)
    int8 matrix (iterations x variables), almost entirely -1 (frozen). Stored as a
    bit-packed mask of the non-frozen cells plus their values in row-major order.
    Lossless, ~1 % of the dense ``.npy``.

Compact pickles (``dump`` / ``load``)
    Any picklable object. Every *variable dict* inside it -- a dict ``{var name:
    {field: bool | int | None, ...}}`` such as a solution, a solution delta or
    ``CppSolveResult.var_sols`` -- is pulled out of the pickle stream into one shared
    columnar table:

        names       every variable name once; 'P[o,s,k]'-style names as prefix id +
                    delta-encoded int columns, others verbatim
        ptr         dict j owns entries ptr[j]:ptr[j+1]
        idx         entry -> name index (delta-encoded)
        sid         entry -> schema id; a schema is the tuple of (key, type) of the
                    value dict, type in b(ool) / i(nt) / n(one), e.g.
                    (('present','b'), ('start','n'), ('end','n'))
        col.<key>   int column per value key (only meaningful where the schema
                    says b/i)

    Values that do not fit a schema (floats, nested objects, ...) are kept verbatim
    in the pickled metadata. Object identity is preserved (``CppSolveResult.var_sols``
    and ``result_dict['var_solutions']`` stay one object), so loading returns exactly
    what was dumped.

``load`` also reads the legacy ``.pkl`` / ``.json`` files, and ``find`` resolves a
path stem to whichever variant exists, so unconverted data keeps working.
Convert old data with ``scripts/compact_results.py``.
"""

import io
import json
import os
import pickle
import re
import zipfile

import numpy as np

FORMAT_VERSION = 1
_EXTRA = 255  # schema id of entries stored verbatim in the metadata
_MAX_SCHEMAS = 255


# --------------------------------------------------------------------------- #
# npz helpers
# --------------------------------------------------------------------------- #
def _write_npz(path, arrays):
    """Atomic np.savez_compressed (no partially written file on crash/interrupt)."""
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def _bytes_array(b):
    return np.frombuffer(b, dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Barcodes
# --------------------------------------------------------------------------- #
def save_barcode(path, matrix, fill=-1):
    """Save a 2-D barcode matrix (mostly ``fill``) as a sparse bit-packed .npz."""
    m = np.asarray(matrix)
    if m.ndim != 2:
        raise ValueError(f"barcode must be 2-D, got shape {m.shape}")
    mask = m != fill
    _write_npz(path, {
        "kind": np.array("barcode"),
        "version": np.array(FORMAT_VERSION),
        "shape": np.array(m.shape, dtype=np.int64),
        "dtype": np.array(m.dtype.str),
        "fill": np.array(fill, dtype=m.dtype),
        "mask": np.packbits(mask, axis=1),
        "values": m[mask],
    })


def load_barcode(path):
    """Load a barcode saved by ``save_barcode`` (or a legacy dense ``.npy``)."""
    if str(path).endswith(".npy"):
        return np.load(path)
    with np.load(path) as z:
        rows, cols = (int(x) for x in z["shape"])
        m = np.full((rows, cols), z["fill"], dtype=np.dtype(str(z["dtype"])))
        mask = np.unpackbits(z["mask"], axis=1, count=cols).view(bool)
        m[mask] = z["values"]
    return m


# --------------------------------------------------------------------------- #
# Variable-dict table
# --------------------------------------------------------------------------- #
def _schema(v):
    """(key, type) tuple of a value dict, or None if it does not fit the table."""
    if type(v) is not dict:
        return None
    out = []
    for k, x in v.items():
        if x is None:
            t = "n"
        elif type(x) is bool:
            t = "b"
        elif type(x) is int and -(2**63) <= x < 2**63:
            t = "i"
        else:
            return None
        out.append((k, t))
    return tuple(out)


_NAME_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\[(-?\d+(?:,-?\d+)*)\]\Z")


def _encode_names(names):
    """Names 'Prefix[i,j,...]' as prefix id + delta-encoded int columns (these
    compress ~100x better than the text); anything else is kept verbatim."""
    n = len(names)
    prefixes, prefix_idx = [], {}
    code = np.full(n, -1, dtype=np.int32)
    arity = np.zeros(n, dtype=np.uint8)
    rows, raw = [], []
    for i, name in enumerate(names):
        m = _NAME_RE.match(name)
        args = m.group(2).split(",") if m else ()
        if (not m or len(args) > 255
                or any(a != str(int(a)) for a in args)):  # '-0', '007' ... -> verbatim
            raw.append(name)
            rows.append(())
            continue
        p = prefix_idx.get(m.group(1))
        if p is None:
            p = prefix_idx[m.group(1)] = len(prefixes)
            prefixes.append(m.group(1))
        code[i] = p
        arity[i] = len(args)
        rows.append(tuple(map(int, args)))
    width = int(arity.max()) if n else 0
    mat = np.zeros((n, width), dtype=np.int64)
    for i, r in enumerate(rows):
        if r:
            mat[i, : len(r)] = r
    mat = np.diff(mat, axis=0, prepend=np.zeros((1, width), np.int64))
    if mat.size and mat.min() >= -(2**31) and mat.max() < 2**31:
        mat = mat.astype(np.int32)
    return {
        "t.n.prefixes": _bytes_array("\n".join(prefixes).encode()),
        "t.n.code": code,
        "t.n.arity": arity,
        "t.n.args": mat,
        "t.n.raw": _bytes_array("\n".join(raw).encode()),
    }


def _decode_names(z):
    def split(key):
        b = z[key].tobytes()
        return b.decode().split("\n") if b else []

    prefixes, raw = split("t.n.prefixes"), iter(split("t.n.raw"))
    code = z["t.n.code"].tolist()
    arity = z["t.n.arity"].tolist()
    args = np.cumsum(z["t.n.args"].astype(np.int64), axis=0).tolist()
    return [
        next(raw) if c < 0 else f"{prefixes[c]}[{','.join(map(str, a[:k]))}]"
        for c, k, a in zip(code, arity, args)
    ]


_SCALARS = (type(None), bool, int, float, str)


def _is_var_dict(obj):
    """{str: {key: scalar}} -- flat value dicts only, so nothing stored verbatim
    can hold a reference shared with the rest of the object."""
    if type(obj) is not dict or not obj:
        return False
    for k, v in obj.items():
        if type(k) is not str or type(v) is not dict or "\n" in k:
            return False
        for x in v.values():
            if type(x) not in _SCALARS:
                return False
    return True


class _Table:
    """Accumulates variable dicts during pickling."""

    def __init__(self):
        self.names, self.name_idx = [], {}
        self.schemas, self.schema_idx = [], {}
        self.ptr = [0]
        self.idx, self.sid = [], []
        self.cols = {}  # key -> {entry position: int}
        self.extra = {}  # entry position -> verbatim value

    def add(self, d):
        names, name_idx = self.names, self.name_idx
        idx, sid, cols = self.idx, self.sid, self.cols
        for name, v in d.items():
            i = name_idx.get(name)
            if i is None:
                i = name_idx[name] = len(names)
                names.append(name)
            pos = len(idx)
            idx.append(i)
            sch = _schema(v)
            s = self.schema_idx.get(sch) if sch is not None else None
            if s is None and sch is not None and len(self.schemas) < _MAX_SCHEMAS:
                s = self.schema_idx[sch] = len(self.schemas)
                self.schemas.append(sch)
            if s is None:
                sid.append(_EXTRA)
                self.extra[pos] = v
                continue
            sid.append(s)
            for k, t in sch:
                if t != "n":
                    cols.setdefault(k, {})[pos] = int(v[k])
        self.ptr.append(len(idx))
        return len(self.ptr) - 2

    def arrays(self):
        n = len(self.idx)
        out = _encode_names(self.names)
        out.update({
            "t.ptr": np.array(self.ptr, dtype=np.int64),
            # delta-encoded: mostly runs of +1 (dict order == name order)
            "t.idx": np.diff(np.array(self.idx, dtype=np.int64), prepend=0),
            "t.sid": np.array(self.sid, dtype=np.uint8),
        })
        col_keys = {}  # value key -> array name
        for i, (k, vals) in enumerate(self.cols.items()):
            col = np.zeros(n, dtype=np.int64)
            col[np.fromiter(vals.keys(), np.int64, len(vals))] = np.fromiter(
                vals.values(), np.int64, len(vals))
            if col.min() >= -(2**31) and col.max() < 2**31:
                col = col.astype(np.int32)
            col_keys[k] = f"t.col.{i}"
            out[col_keys[k]] = col
        return out, {"schemas": self.schemas, "extra": self.extra,
                     "col_keys": col_keys}


def _decode_table(z, meta):
    names = _decode_names(z)
    ptr = z["t.ptr"].tolist()
    idx = np.cumsum(z["t.idx"]).tolist()
    sid = z["t.sid"].tolist()
    cols = {k: z[a].tolist() for k, a in meta["col_keys"].items()}
    schemas, extra = meta["schemas"], meta["extra"]
    # per-schema field accessors: (key, column or None, is_bool)
    plans = [[(k, cols.get(k) if t != "n" else None, t == "b") for k, t in sch]
             for sch in schemas]
    dicts = []
    for j in range(len(ptr) - 1):
        d = {}
        for p in range(ptr[j], ptr[j + 1]):
            s = sid[p]
            if s == _EXTRA:
                d[names[idx[p]]] = extra[p]
                continue
            v = {}
            for k, col, is_bool in plans[s]:
                if col is None:
                    v[k] = None
                elif is_bool:
                    v[k] = bool(col[p])
                else:
                    v[k] = col[p]
            d[names[idx[p]]] = v
        dicts.append(d)
    return dicts


class _Pickler(pickle.Pickler):
    def __init__(self, f, table):
        super().__init__(f, protocol=pickle.HIGHEST_PROTOCOL)
        self.table = table
        self.seen = {}  # id(obj) -> (pid, obj); keeps shared references shared

    def persistent_id(self, obj):
        if type(obj) is not dict:
            return None
        hit = self.seen.get(id(obj))
        if hit is not None:
            return hit[0]
        if not _is_var_dict(obj):
            return None
        pid = self.table.add(obj)
        self.seen[id(obj)] = (pid, obj)
        return pid


class _Unpickler(pickle.Unpickler):
    def __init__(self, f, dicts):
        super().__init__(f)
        self.dicts = dicts

    def persistent_load(self, pid):
        return self.dicts[pid]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def dump(obj, path):
    """Write ``obj`` as a compact pickle (.npz). See module docstring."""
    table = _Table()
    buf = io.BytesIO()
    _Pickler(buf, table).dump(obj)
    arrays, meta = table.arrays()
    meta["version"] = FORMAT_VERSION
    arrays["kind"] = np.array("compact_pickle")
    arrays["meta"] = _bytes_array(pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL))
    arrays["pickle"] = _bytes_array(buf.getvalue())
    _write_npz(path, arrays)


def load(path):
    """Load a compact pickle (.npz), a legacy pickle (.pkl) or JSON (.json)."""
    path = str(path)
    if path.endswith(".json"):
        with open(path) as f:
            return json.load(f)
    if path.endswith(".npz") and zipfile.is_zipfile(path):
        with np.load(path, allow_pickle=False) as z:
            if str(z["kind"]) == "barcode":
                raise ValueError(f"{path} is a barcode; use load_barcode")
            meta = pickle.loads(z["meta"].tobytes())
            dicts = _decode_table(z, meta)
            return _Unpickler(io.BytesIO(z["pickle"].tobytes()), dicts).load()
    with open(path, "rb") as f:
        return pickle.load(f)


LEGACY_EXTS = (".pkl", ".json", ".npy")


def find(stem):
    """Existing file for a path given with or without extension: the compact
    ``.npz`` if present, else the legacy ``.pkl`` / ``.json`` / ``.npy``; None if
    none exists."""
    stem = str(stem)
    for ext in (".npz",) + LEGACY_EXTS:
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    for ext in (".npz",) + LEGACY_EXTS:
        if os.path.isfile(stem + ext):
            return stem + ext
    return None


def find_all(stem_pattern):
    """Sorted existing files matching a glob over stems (no extension), one per
    stem, preferring ``.npz`` (e.g. ``'<dir>/cp_intermediate_records_*'``)."""
    import glob

    stems = {
        p[: -len(ext)]
        for ext in (".npz",) + LEGACY_EXTS
        for p in glob.glob(stem_pattern + ext)
    }
    return sorted(find(s) for s in stems)

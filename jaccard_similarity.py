from collections import defaultdict

# ============================================================
# (Weighted) Jaccard similarity
# ============================================================


def compute_demand_count(
    orders_req: dict[int, list[int]],
    K: list[int],
) -> dict[int, int]:
    """Count how many orders need each SKU."""
    count: dict[int, int] = defaultdict(int)
    for o in orders_req:
        for k in orders_req[o]:
            count[k] += 1
    return dict(count)


def compute_jaccard_weighted_and_unweighted(
    o1: int,
    o2: int,
    orders_req: dict[int, list[int]],
    rt: dict[int, int],
    rt_ret: dict[int, int],
    normalize: bool = True,
) -> float:
    """Retrieval-time-weighted Jaccard similarity between two orders.

    J_w(o, o') = Σ_{k ∈ R_o ∩ R_o'} (rt[k]+rt_ret[k])
                 / Σ_{k ∈ R_o ∪ R_o'} (rt[k]+rt_ret[k])

    Returns 0.0 when both orders are empty or have no union.
    """
    set1 = set(orders_req[o1])
    set2 = set(orders_req[o2])
    intersection = set1 & set2
    union = set1 | set2

    if len(union) > 0:
        unweighted_jaccard = (
            len(intersection) / len(union) if normalize else len(intersection)
        )
    else:
        unweighted_jaccard = 0.0

    union_weight = sum(rt[k] + rt_ret[k] for k in union)
    intersection_weight = sum(rt[k] + rt_ret[k] for k in intersection)

    if union_weight > 0:
        weighted_jaccard = (
            intersection_weight / union_weight if normalize else intersection_weight
        )
    else:
        weighted_jaccard = 0.0

    return weighted_jaccard, unweighted_jaccard


def build_similarity_matrix(
    O: list[int],
    orders_req: dict[int, list[int]],
    rt: dict[int, int],
    rt_ret: dict[int, int],
    normalize: bool = True,
) -> dict[tuple[int, int], float]:
    """Compute pairwise weighted Jaccard for all order pairs (upper triangle).

    Returns {(o1, o2): similarity} where o1 < o2 in the O list index sense.
    """
    sim_weighted: dict[tuple[int, int], float] = {}
    sim_unweighted: dict[tuple[int, int], float] = {}

    for i in range(len(O)):
        for j in range(i + 1, len(O)):
            o1, o2 = O[i], O[j]
            key = (o1, o2) if o1 < o2 else (o2, o1)
            sim_weighted[key], sim_unweighted[key] = (
                compute_jaccard_weighted_and_unweighted(
                    o1, o2, orders_req, rt, rt_ret, normalize=normalize
                )
            )
    return sim_weighted, sim_unweighted


# ============================================================
#
# ============================================================

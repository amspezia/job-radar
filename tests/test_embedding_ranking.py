from math import sqrt
from uuid import UUID

import numpy as np
import pytest

from eval.embedding.ranking import compare_rankings, cosine_rank


def uid(n: int) -> UUID:
    return UUID(int=n)


def brute_force(matrix, ids, query):
    """Reference cosine ranking written without numpy."""
    rows = [[float(x) for x in row] for row in matrix]
    q = [float(x) for x in query]
    q_norm = sqrt(sum(x * x for x in q))
    scored = []
    for doc_id, row in zip(ids, rows, strict=True):
        row_norm = sqrt(sum(x * x for x in row))
        denom = row_norm * q_norm
        dot = sum(x * y for x, y in zip(row, q, strict=True))
        scored.append((doc_id, dot / denom if denom > 0 else 0.0))
    return sorted(scored, key=lambda pair: (-pair[1], pair[0]))


def test_cosine_rank_matches_brute_force():
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(40, 16))
    query = rng.normal(size=16)
    ids = [uid(i) for i in range(40)]

    got = cosine_rank(matrix, ids, query)
    want = brute_force(matrix, ids, query)

    assert [doc_id for doc_id, _ in got] == [doc_id for doc_id, _ in want]
    assert [score for _, score in got] == pytest.approx([score for _, score in want])


def test_cosine_rank_is_scale_invariant():
    rng = np.random.default_rng(11)
    unit = rng.normal(size=(12, 8))
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    scaled = unit * rng.uniform(0.1, 9.0, size=(12, 1))
    ids = [uid(i) for i in range(12)]
    # A mean of unit vectors is not unit norm — that is what the harness feeds in.
    query = unit[0] + unit[1] + unit[2]

    got = cosine_rank(scaled, ids, query)
    want = cosine_rank(unit, ids, query)

    assert [doc_id for doc_id, _ in got] == [doc_id for doc_id, _ in want]
    assert [score for _, score in got] == pytest.approx([score for _, score in want])


def test_cosine_rank_breaks_identical_vector_ties_by_id():
    matrix = np.tile(np.array([0.3, 0.4, 0.5]), (5, 1))
    ids = [uid(n) for n in (9, 2, 7, 1, 4)]

    ranked = cosine_rank(matrix, ids, np.array([0.3, 0.4, 0.5]))

    assert [doc_id for doc_id, _ in ranked] == sorted(ids)
    assert [score for _, score in ranked] == pytest.approx([1.0] * 5)


def test_cosine_rank_zero_vectors_score_zero():
    matrix = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    ids = [uid(3), uid(2), uid(1)]

    ranked = cosine_rank(matrix, ids, np.array([1.0, 0.0]))

    assert ranked == [(uid(2), 1.0), (uid(1), 0.0), (uid(3), 0.0)]
    assert not np.isnan([score for _, score in ranked]).any()


def test_cosine_rank_zero_query_scores_zero():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]])

    ranked = cosine_rank(matrix, [uid(2), uid(1)], np.array([0.0, 0.0]))

    assert ranked == [(uid(1), 0.0), (uid(2), 0.0)]


@pytest.mark.parametrize(
    ("matrix", "ids", "query", "message"),
    [
        (np.zeros((2, 2, 2)), [uid(1), uid(2)], np.zeros(2), "2-d"),
        (np.zeros((2, 2)), [uid(1), uid(2)], np.zeros((1, 2)), "1-d"),
        (np.zeros((2, 2)), [uid(1)], np.zeros(2), "ids"),
        (np.zeros((2, 3)), [uid(1), uid(2)], np.zeros(2), "3-d but query is 2-d"),
    ],
)
def test_cosine_rank_validates_shapes(matrix, ids, query, message):
    with pytest.raises(ValueError, match=message):
        cosine_rank(matrix, ids, query)


def ranking(*pairs):
    return [(uid(n), score) for n, score in pairs]


def test_compare_rankings_equal():
    a = ranking((1, 0.9), (2, 0.8), (3, 0.7), (4, 0.1))

    result = compare_rankings(a, list(a), 3, 1e-5)

    assert result.ok
    assert result.reasons == []


def test_compare_rankings_accepts_permuted_ties():
    a = ranking((1, 0.9), (2, 0.9), (3, 0.9), (4, 0.1))
    b = ranking((3, 0.9), (1, 0.9), (2, 0.9), (4, 0.1))

    assert compare_rankings(a, b, 4, 1e-5).ok


def test_compare_rankings_rejects_swapped_non_ties():
    a = ranking((1, 0.9), (2, 0.5), (3, 0.1))
    b = ranking((2, 0.9), (1, 0.5), (3, 0.1))

    result = compare_rankings(a, b, 3, 1e-5)

    assert not result.ok
    assert any("id set differs in positions 0-0" in reason for reason in result.reasons)


def test_compare_rankings_rejects_score_drift():
    a = ranking((1, 0.9), (2, 0.5), (3, 0.1))
    b = ranking((1, 0.9), (2, 0.4), (3, 0.1))

    result = compare_rankings(a, b, 3, 1e-5)

    assert not result.ok
    assert any("score drift at position 1" in reason for reason in result.reasons)


def test_compare_rankings_rejects_set_mismatch():
    a = ranking((1, 0.9), (2, 0.9), (3, 0.2))
    b = ranking((1, 0.9), (4, 0.9), (3, 0.2))

    result = compare_rankings(a, b, 3, 1e-5)

    assert not result.ok
    assert any("only in b" in reason for reason in result.reasons)


def test_compare_rankings_tolerates_a_tie_chain_crossing_k():
    # Positions 1 and 2 are a tie chain; k=2 cuts it, so b may hold either member at 1.
    a = ranking((1, 0.9), (2, 0.5), (3, 0.5), (4, 0.1))
    b = ranking((1, 0.9), (3, 0.5), (2, 0.5), (4, 0.1))

    assert compare_rankings(a, b, 2, 1e-5).ok


def test_compare_rankings_rejects_a_foreign_id_in_the_crossing_chain():
    # b promotes doc 4 into the tie chain at a score b agrees with but a scores far lower:
    # the position-wise check cannot see it, the chain band can.
    a = ranking((1, 1.0), (2, 0.9), (3, 0.9), (4, 0.5))
    b = ranking((1, 1.0), (4, 0.9))

    result = compare_rankings(a, b, 2, 0.01)

    assert not result.ok
    assert any("outside the tie chain 1-2" in reason for reason in result.reasons)


def test_compare_rankings_rejects_a_short_ranking():
    a = ranking((1, 0.9), (2, 0.5), (3, 0.1))

    result = compare_rankings(a, a[:2], 3, 1e-5)

    assert not result.ok
    assert result.reasons == ["ranking shorter than k=3: len(a)=3, len(b)=2"]


def test_compare_rankings_caps_its_reasons():
    a = ranking(*[(n, 1.0 - n / 100) for n in range(20)])
    b = ranking(*[(19 - n, 0.0) for n in range(20)])

    result = compare_rankings(a, b, 20, 1e-5)

    assert not result.ok
    assert len(result.reasons) == 10  # five score drifts, five id-set mismatches

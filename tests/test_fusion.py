from uuid import UUID

from job_radar.retrieval.fusion import reciprocal_rank_fusion

# Stable, readable ids for assertions.
A, B, C, D = (UUID(int=i) for i in range(4))


def _ids(fused: list[tuple[UUID, float]]) -> list[UUID]:
    return [job_id for job_id, _ in fused]


def test_fuses_two_rankings_rewarding_cross_list_agreement() -> None:
    # B is #1 in neither list but ranks highly in both, so it should win.
    fts = [(A, 9.0), (B, 8.0), (C, 7.0)]  # ranks 1, 2, 3
    vector = [(B, 0.9), (D, 0.8), (A, 0.7)]  # ranks 1, 2, 3

    fused = reciprocal_rank_fusion([fts, vector], k=60)

    assert _ids(fused) == [B, A, D, C]
    # Hand-computed against 1/(k+rank): B = 1/61 + 1/62, A = 1/61 + 1/63.
    by_id = dict(fused)
    assert by_id[B] == 1 / 61 + 1 / 62
    assert by_id[A] == 1 / 61 + 1 / 63


def test_ignores_component_scores_and_uses_only_rank() -> None:
    # Wildly different score scales must not matter — only positions do.
    # A is rank 1 in both lists, B is rank 2 in both, despite the inverted
    # raw scores; if scores leaked in, B would be pulled up.
    fts = [(A, 0.001), (B, 1000.0)]
    vector = [(A, 0.001), (B, 1000.0)]

    by_id = dict(reciprocal_rank_fusion([fts, vector]))

    assert by_id[A] == 1 / 61 + 1 / 61  # rank 1 in both
    assert by_id[B] == 1 / 62 + 1 / 62  # rank 2 in both
    assert by_id[A] > by_id[B]  # higher-ranked wins regardless of raw score


def test_single_ranking_preserves_order() -> None:
    ranking = [(A, 5.0), (B, 4.0), (C, 3.0)]

    fused = reciprocal_rank_fusion([ranking])

    assert _ids(fused) == [A, B, C]


def test_limit_truncates_after_fusion() -> None:
    fts = [(A, 9.0), (B, 8.0), (C, 7.0)]
    vector = [(B, 0.9), (A, 0.8), (C, 0.7)]

    fused = reciprocal_rank_fusion([fts, vector], limit=2)

    assert len(fused) == 2
    assert _ids(fused) == [A, B]  # A and B both appear twice near the top


def test_ties_break_deterministically_on_id() -> None:
    # A and B each appear once at rank 1 in separate lists -> identical scores.
    fused = reciprocal_rank_fusion([[(B, 1.0)], [(A, 1.0)]])

    by_id = dict(fused)
    assert by_id[A] == by_id[B]
    assert _ids(fused) == [A, B]  # tie broken by str(id): A (int=0) < B (int=1)


def test_larger_k_flattens_rank_one_dominance() -> None:
    ranking = [(A, 9.0), (B, 8.0)]

    small_k = dict(reciprocal_rank_fusion([ranking], k=1))
    large_k = dict(reciprocal_rank_fusion([ranking], k=1000))

    # Bigger k shrinks the gap between rank 1 and rank 2.
    assert small_k[A] - small_k[B] > large_k[A] - large_k[B]


def test_empty_input_returns_empty() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []

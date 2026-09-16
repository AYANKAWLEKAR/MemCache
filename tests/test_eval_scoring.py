"""Unit tests for evals.scoring — pure functions, no stores or models."""

from evals.scoring import Aggregate, aggregate, count_tokens, fact_coverage, normalize


class TestNormalize:
    def test_casefolds_and_collapses_whitespace(self):
        assert normalize("  DuplicateColumn:\n  user_id  ") == "duplicatecolumn: user_id"

    def test_empty_and_none_are_empty(self):
        assert normalize("") == ""
        assert normalize(None) == ""


class TestFactCoverage:
    def test_all_present(self):
        cov = fact_coverage("We decided to use Rust for the backend.", ["rust", "backend"])
        assert cov.fraction == 1.0
        assert cov.missing == ()

    def test_partial(self):
        cov = fact_coverage("We decided to use Rust.", ["rust", "clickhouse"])
        assert cov.present == ("rust",)
        assert cov.missing == ("clickhouse",)
        assert cov.fraction == 0.5

    def test_match_is_case_insensitive_and_whitespace_normalized(self):
        text = "error was  DuplicateColumn:   column user_id already exists"
        cov = fact_coverage(text, ["duplicatecolumn: column USER_ID already exists"])
        assert cov.fraction == 1.0

    def test_no_required_facts_scores_one(self):
        assert fact_coverage("anything", []).fraction == 1.0

    def test_empty_answer_scores_zero(self):
        assert fact_coverage("", ["rust"]).fraction == 0.0


class TestCountTokens:
    def test_matches_retrieval_tier_counting(self):
        # The eval's token metric must mean the same thing as the retrieval
        # tier's max_tokens budget. Pin the mirrored implementation to the
        # app's own counter on representative texts.
        from app.services.retrieval import _count_tokens

        samples = ["", "one two three", "DuplicateColumn: column user_id already exists\n" * 40]
        for text in samples:
            assert count_tokens(text) == _count_tokens(text)

    def test_monotone_in_length(self):
        assert count_tokens("word " * 200) > count_tokens("word " * 10)


class TestAggregate:
    def test_mean_min_max_n(self):
        agg = aggregate([0.0, 0.5, 1.0])
        assert agg == Aggregate(mean=0.5, minimum=0.0, maximum=1.0, n=3)

    def test_empty_returns_none(self):
        assert aggregate([]) is None

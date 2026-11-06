"""Tests for the movie EDA helper functions."""

import eda


def test_parse_list():
    """Check valid and invalid list text."""
    assert eda.parse_list("[{'name': 'Drama'}]") == [{"name": "Drama"}]
    assert eda.parse_list("not a list") == []


def test_number():
    """Check positive, missing, and non-positive values."""
    assert eda.number("20") == 20.0
    assert eda.number("0") is None
    assert eda.number("") is None


def test_prepare_metadata_and_joins():
    """Check ID validation and that both joins match on movie ID."""
    rows = [{"id": "10", "budget": "100", "revenue": "200",
             "genres": "[{'name': 'Drama'}]", "title": "Film"},
            {"id": "bad", "budget": "1", "revenue": "1", "genres": "[]"}]
    movies, duplicates = eda.prepare_metadata(rows)
    assert len(movies) == 1
    assert duplicates == 0
    cast, cast_ids = eda.merge_cast(
        movies, [{"id": "10", "cast": "[{'name': 'Actor'}]"}])
    assert cast[0]["cast_size"] == 1
    assert cast_ids == {"10"}
    keywords, keyword_ids = eda.merge_keywords(
        movies, [{"id": "10", "keywords": "[{'name': 'Time Travel'}]"}])
    assert keywords[0]["keywords"] == ["time travel"]
    assert keyword_ids == {"10"}


def test_summarize():
    """Check summary handles ordinary and empty data."""
    result = eda.summarize([1, 2, 3, 4])
    assert "n=4" in result
    assert "median=2.50" in result
    assert eda.summarize([]) == "No usable values"


def main():
    """Run the tests."""
    test_parse_list()
    test_number()
    test_prepare_metadata_and_joins()
    test_summarize()
    print("All tests passed.")


if __name__ == "__main__":
    main()

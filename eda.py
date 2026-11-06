"""Explore movie revenue, budget, cast size, genres, and keywords."""

import ast
import csv
import os
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DATA_DIR, "eda_output")


def read_csv(filename):
    """Read a CSV file.

    Returns: list[dict[str, str]]: Rows keyed by the CSV column names.
    """
    with open(os.path.join(DATA_DIR, filename), encoding="utf-8",
              newline="") as data_file:
        return list(csv.DictReader(data_file))


def parse_list(value):
    """Parse Python-style list text.

    Returns: list: Parsed list; malformed or non-list input returns [].
    """
    try:
        result = ast.literal_eval(value)
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError, TypeError):
        return []


def number(value):
    """Convert a positive numeric string to float.

    Returns: float | None: Positive value, or None for invalid input.
    """
    try:
        result = float(value)
        if result <= 0:
            return None
        return result
    except (ValueError, TypeError):
        return None


def prepare_metadata(rows):
    """Keep unique numeric movie IDs and selected metadata.

    Returns: tuple[dict, int]: ID-to-movie dictionary and duplicate-ID count.
    """
    movies = {}
    duplicate_ids = 0
    for row in rows:
        movie_id = row.get("id", "").strip()
        if not movie_id.isdigit():
            continue
        if movie_id in movies:
            duplicate_ids += 1
            continue
        budget = number(row.get("budget"))
        revenue = number(row.get("revenue"))
        genres = [item.get("name", "") for item in parse_list(
            row.get("genres", "")) if isinstance(item, dict)]
        movies[movie_id] = {
            "budget": budget,
            "revenue": revenue,
            "genres": genres,
            "title": row.get("title", ""),
        }
    return movies, duplicate_ids


def merge_cast(movies, credits):
    """Join metadata and credits by movie ID and count listed cast members.

    Returns: tuple[list[dict], set[str]]: Joined rows and matched movie IDs.
    """
    merged = []
    matched = set()
    for row in credits:
        movie_id = row.get("id", "").strip()
        if movie_id not in movies:
            continue
        cast = parse_list(row.get("cast", ""))
        movie = movies[movie_id]
        merged.append({"id": movie_id, "cast_size": len(cast),
                       "budget": movie["budget"],
                       "revenue": movie["revenue"]})
        matched.add(movie_id)
    return merged, matched


def merge_keywords(movies, keywords):
    """Join metadata and keywords by ID; lowercase and normalize whitespace.

    Returns: tuple[list[dict], set[str]]: Joined rows and matched movie IDs.
    """
    merged = []
    matched = set()
    for row in keywords:
        movie_id = row.get("id", "").strip()
        if movie_id not in movies:
            continue
        terms = [re.sub(r"\s+", " ", item.get("name", "").strip().lower())
                 for item in parse_list(row.get("keywords", ""))
                 if isinstance(item, dict) and item.get("name", "").strip()]
        movie = movies[movie_id]
        merged.append({"id": movie_id, "keywords": terms,
                       "revenue": movie["revenue"],
                       "budget": movie["budget"]})
        matched.add(movie_id)
    return merged, matched


def summarize(values):
    """Summarize numeric values with count, mean, SD, and five-number summary.

    Returns: str: Formatted statistics, or a message when no values are usable.
    """
    values = sorted(value for value in values if value is not None)
    if not values:
        return "No usable values"
    count = len(values)
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count

    def percentile(fraction):
        index = (count - 1) * fraction
        lower = int(index)
        upper = min(lower + 1, count - 1)
        difference = values[upper] - values[lower]
        return values[lower] + difference * (index - lower)

    return ("n={:,}, mean={:,.2f}, SD={:,.2f}, min={:,.2f}, "
            "Q1={:,.2f}, median={:,.2f}, Q3={:,.2f}, max={:,.2f}".format(
                count, mean, variance ** 0.5, values[0], percentile(.25),
                percentile(.5), percentile(.75), values[-1]))


def save_plot(filename, title, caption, plot_function):
    """Save a titled and captioned plot in OUTPUT_DIR.

    Returns: None. The figure is written as a PNG file.
    """
    plt.figure(figsize=(8, 5))
    plot_function()
    plt.title(title)
    plt.tight_layout(rect=[0, .08, 1, 1])
    plt.figtext(.02, .015, caption, wrap=True, fontsize=9)
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=160)
    plt.close()


def analyze():
    """Run joins, statistics, and six plots for the three EDA datasets.

    Returns: dict[str, object]: Counts, statistics, correlations, frequencies.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metadata_rows = read_csv("movies_metadata.csv")
    credits_rows = read_csv("credits.csv")
    keywords_rows = read_csv("keywords.csv")
    movies, duplicate_ids = prepare_metadata(metadata_rows)
    cast_rows, cast_ids = merge_cast(movies, credits_rows)
    keyword_rows, keyword_ids = merge_keywords(movies, keywords_rows)
    valid_finance = [movie for movie in movies.values()
                     if movie["budget"] and movie["revenue"]]
    budget = [movie["budget"] for movie in valid_finance]
    revenue = [movie["revenue"] for movie in valid_finance]
    corr_budget = spearmanr(budget, revenue)
    cast_finance = [row for row in cast_rows
                    if row["cast_size"] > 0 and row["revenue"]]
    corr_cast = spearmanr([row["cast_size"] for row in cast_finance],
                          [row["revenue"] for row in cast_finance])
    genres = Counter(genre for movie in valid_finance
                     for genre in movie["genres"])
    keywords = Counter(term for row in keyword_rows
                       for term in row["keywords"])
    selected_keywords = [term for term, count in keywords.most_common(8)
                         if count >= 100]
    keyword_revenues = {term: [row["revenue"] for row in keyword_rows
                               if row["revenue"] and term in row["keywords"]]
                        for term in selected_keywords}

    def budget_revenue_plot():
        plt.scatter(budget, revenue, alpha=.18, s=10, label="Movies")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Budget (USD, log scale)")
        plt.ylabel("Revenue (USD, log scale)")
        plt.legend()

    save_plot("budget_revenue.png", "Movie Budget and Revenue",
              "Caption: Usable positive values appear on log scales; points "
              "show association, not causation.", budget_revenue_plot)

    def genre_plot():
        top = genres.most_common(10)
        plt.bar([item[0] for item in top], [item[1] for item in top],
                label="Movies")
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Movie appearances")
        plt.legend()

    save_plot("genre_counts.png", "Genres Among Movies with Valid Finances",
              "Caption: Genre counts can exceed the number of movies because"
              " each movie may have multiple genres.", genre_plot)

    def cast_revenue_plot():
        plt.scatter([row["cast_size"] for row in cast_finance],
                    [row["revenue"] for row in cast_finance],
                    alpha=.18, s=10, label="Movies")
        plt.yscale("log")
        plt.xlabel("Number of listed cast members")
        plt.ylabel("Revenue (USD, log scale)")
        plt.legend()

    save_plot("cast_revenue.png", "Listed Cast Size and Revenue",
              "Caption: Revenue varies widely at each cast size; only positive"
              " revenue and nonempty parsed casts are included.",
              cast_revenue_plot)

    def cast_histogram():
        plt.hist([row["cast_size"] for row in cast_rows], bins=30,
                 label="Movies")
        plt.xlabel("Number of listed cast members")
        plt.ylabel("Movie count")
        plt.legend()

    save_plot("cast_size.png", "Distribution of Listed Cast Size",
              "Caption: The distribution describes cast entries in the credits"
              " file and does not establish that every cast list is complete.",
              cast_histogram)

    def keyword_plot():
        names = list(selected_keywords)
        medians = [sorted(keyword_revenues[name])[
            len(keyword_revenues[name]) // 2] for name in names]
        plt.bar(names, medians, label="Median revenue")
        plt.xticks(rotation=35, ha="right")
        plt.yscale("log")
        plt.ylabel("Median revenue (USD, log scale)")
        plt.legend()

    save_plot("keyword_revenue.png", "Revenue for Frequent Keywords",
              "Caption: Median revenue for keywords in at least 100 "
              "joined rows; groups overlap because movies have multiple "
              "terms.",
              keyword_plot)

    def keyword_frequency_plot():
        top = keywords.most_common(10)
        plt.bar([item[0] for item in top], [item[1] for item in top],
                label="Keyword appearances")
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Movie appearances")
        plt.legend()

    save_plot("keyword_frequency.png", "Most Frequent Keywords",
              "Caption: Keyword names were lowercased and spaces normalized "
              "before counting.", keyword_frequency_plot)

    return {
        "metadata_rows": len(metadata_rows), "metadata_columns": 24,
        "credits_rows": len(credits_rows), "credits_columns": 3,
        "keywords_rows": len(keywords_rows), "keywords_columns": 2,
        "valid_ids": len(movies), "duplicate_ids": duplicate_ids,
        "cast_rows": len(cast_rows), "cast_matched": len(cast_ids),
        "keyword_rows": len(keyword_rows), "keyword_matched": len(keyword_ids),
        "valid_finance": len(valid_finance),
        "budget_summary": summarize(budget),
        "revenue_summary": summarize(revenue),
        "cast_summary": summarize([row["cast_size"] for row in cast_rows]),
        "budget_rho": corr_budget.statistic, "budget_p": corr_budget.pvalue,
        "cast_rho": corr_cast.statistic, "cast_p": corr_cast.pvalue,
        "genres": genres.most_common(10),
        "keywords": [(term, len(keyword_revenues[term]),
                      summarize(keyword_revenues[term]))
                     for term in selected_keywords],
    }


def write_findings(findings):
    """Write findings to eda_output/findings.txt.

    Args:
        findings: dict[str, object] of report-ready EDA results.

    Returns: None.
    """
    path = os.path.join(OUTPUT_DIR, "findings.txt")
    with open(path, "w", encoding="utf-8") as out:
        for key, value in findings.items():
            out.write("{}: {}\n".format(key, value))


def main():
    """Run EDA and print findings.

    Returns: None. Output files are saved in eda_output/.
    """
    findings = analyze()
    write_findings(findings)
    print("EDA complete. Results saved to {}".format(OUTPUT_DIR))
    for key, value in findings.items():
        print("{}: {}".format(key, value))


if __name__ == "__main__":
    main()

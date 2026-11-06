# CSE 163 Movie Project: Exploratory Data Analysis

This project explores associations between movie budget, revenue, genres, cast size, and keywords using The Movies Dataset.

## Files

- `eda.py`: runnable analysis; creates six plots and a findings summary in `eda_output/`.
- `test_eda.py`: small tests for parsing, cleaning, joins, and summary calculations.
- `eda.pdf`: current exploratory analysis report.

The original CSV files are not included in this repository because they are large. Download The Movies Dataset from Kaggle and put `movies_metadata.csv`, `credits.csv`, and `keywords.csv` in this folder.

## Run

Install the required libraries:

```text
python3 -m pip install matplotlib scipy
```

Run the tests and analysis:

```text
python3 test_eda.py
python3 eda.py
```

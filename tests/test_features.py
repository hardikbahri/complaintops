import pandas as pd

from complaintops.features import FEATURE_COLUMNS, build_features, text_features


def test_text_counts_on_three_examples():
    f = text_features(pd.Series(["I paid $50 and $20", "XXXX XXXX owes me", ""]))
    assert f["n_dollar"].tolist() == [2, 0, 0]
    assert f["n_xxxx"].tolist() == [0, 2, 0]
    assert f["n_words"].tolist() == [5, 4, 0]
    assert f["n_chars"].tolist() == [18, 17, 0]


def test_dollar_pattern_handles_space_and_ignores_plain_numbers():
    f = text_features(pd.Series(["cost $ 5 total", "2023 and 500 items"]))
    assert f["n_dollar"].tolist() == [1, 0]


def test_build_features_adds_day_of_week_and_no_leaky_columns():
    df = pd.DataFrame({
        "narrative": ["a b", "XXXX"],
        "date": pd.to_datetime(["2023-01-02", "2023-01-07"]),  # Monday, Saturday
        "product": ["x", "y"],
        "issue": ["x", "y"],
    })
    f = build_features(df)
    assert list(f.columns) == FEATURE_COLUMNS
    assert f["day_of_week"].tolist() == [0, 5]
    assert not {"product", "issue", "label"} & set(f.columns)

import pandas as pd
import pytest

from complaintops.data import apply_other, split_by_date, top_labels


def norm(s):
    return s.str.lower().str.split().str.join(" ")


def test_split_is_chronological():
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=100), "label": "a"})
    tr, va, te = split_by_date(df.sample(frac=1, random_state=0))
    assert tr["date"].max() <= va["date"].min() <= va["date"].max() <= te["date"].min()
    assert (len(tr), len(va), len(te)) == (70, 15, 15)


def test_top_labels_use_train_only_and_rest_is_other():
    train = pd.DataFrame({"label": ["a"] * 5 + ["b"] * 3 + ["c"]})
    other = pd.DataFrame({"label": ["a", "c", "z"]})
    keep = top_labels(train, n=2)
    assert keep == {"a", "b"}
    assert apply_other(other, keep)["label"].tolist() == ["a", "Other", "Other"]


@pytest.fixture(scope="module")
def splits():
    from pathlib import Path
    if not Path("data/processed/train.parquet").exists():
        pytest.skip("run `python -m complaintops.data` first")
    return {k: pd.read_parquet(f"data/processed/{k}.parquet") for k in ["train", "val", "test"]}


def test_no_narrative_in_two_splits(splits):
    keys = {k: set(norm(v["narrative"])) for k, v in splits.items()}
    assert not keys["train"] & keys["val"]
    assert not keys["train"] & keys["test"]
    assert not keys["val"] & keys["test"]

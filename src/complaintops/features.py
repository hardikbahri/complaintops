"""Feature engineering for the XGBoost model. Step 4.

Every feature here is known when a complaint arrives (leakage check):
text statistics and the received date. Deliberately NOT used:
company_response / timely_response (only exist after processing), product / issue (the target),
submitted_via (constant "Web" in this data, so no information).
"""
import re

import pandas as pd

DOLLAR = re.compile(r"\$\s?\d")

FEATURE_COLUMNS = ["n_chars", "n_words", "n_xxxx", "n_dollar", "day_of_week"]


def text_features(narratives: pd.Series) -> pd.DataFrame:
    """Numeric signals that TF-IDF loses or only sees indirectly."""
    return pd.DataFrame({
        "n_chars": narratives.str.len(),
        "n_words": narratives.str.split().str.len(),
        # Redaction tokens. Redacted dates look like "XX/XX/2023" and are not counted.
        "n_xxxx": narratives.str.count("XXXX"),
        "n_dollar": narratives.str.count(DOLLAR),
    })


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df needs the columns `narrative` and `date`. Returns FEATURE_COLUMNS."""
    out = text_features(df["narrative"])
    out["day_of_week"] = pd.to_datetime(df["date"]).dt.day_of_week
    return out[FEATURE_COLUMNS]

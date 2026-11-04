"""Load, clean and split the CFPB complaints. Step 2."""
import pandas as pd

MIN_LEN=10

PRODUCT_MAP = {
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
    "Credit reporting or other personal consumer reports": "Credit reporting",
    "Credit card or prepaid card": "Credit card",
    "Prepaid card": "Credit card",
    "Bank account or service": "Checking or savings account",
    "Payday loan, title loan, or personal loan": "Payday/personal loan",
    "Payday loan, title loan, personal loan, or advance loan": "Payday/personal loan",
    "Payday loan": "Payday/personal loan",
    "Money transfers": "Money transfer, virtual currency, or money service",
    "Virtual currency": "Money transfer, virtual currency, or money service",
}

# The CFPB reworded some issue menu options over the years; merge old -> new.
ISSUE_MAP = {
    "Problem with a company's investigation into an existing problem":
        "Problem with a credit reporting company's investigation into an existing problem",
    "Incorrect information on credit report": "Incorrect information on your report",
    "Dealing with my lender or servicer": "Dealing with your lender or servicer",
    "Cont'd attempts collect debt not owed": "Attempts to collect debt not owed",
}

# The CFPB replaced its issue menu on ~2017-04-24. Map retired options to the
# closest modern one, per product. (product, old issue) -> new issue.
# Mortgage and checking mappings are approximate (old options were broader/different).
PRODUCT_ISSUE_MAP = {
    ("Credit reporting", "Credit reporting company's investigation"):
        "Problem with a credit reporting company's investigation into an existing problem",
    ("Credit reporting", "Problem with a company's investigation into an existing issue"):
        "Problem with a credit reporting company's investigation into an existing problem",
    ("Credit reporting", "Unable to get credit report/credit score"):
        "Unable to get your credit report or credit score",
    ("Credit reporting", "Improper use of my credit report"): "Improper use of your report",
    ("Credit reporting", "Credit monitoring or identity protection"):
        "Credit monitoring or identity theft protection services",
    ("Credit reporting", "Identity theft protection or other monitoring services"):
        "Credit monitoring or identity theft protection services",
    ("Mortgage", "Loan servicing, payments, escrow account"): "Trouble during payment process",
    ("Mortgage", "Loan modification,collection,foreclosure"): "Struggling to pay mortgage",
    ("Mortgage", "Application, originator, mortgage broker"):
        "Applying for a mortgage or refinancing an existing mortgage",
    ("Mortgage", "Credit decision / Underwriting"):
        "Applying for a mortgage or refinancing an existing mortgage",
    ("Mortgage", "Settlement process and costs"): "Closing on a mortgage",
    ("Checking or savings account", "Account opening, closing, or management"): "Managing an account",
    ("Checking or savings account", "Deposits and withdrawals"): "Managing an account",
    ("Checking or savings account", "Using a debit or ATM card"): "Managing an account",
    ("Checking or savings account", "Making/receiving payments, sending money"): "Managing an account",
    ("Checking or savings account", "Problems caused by my funds being low"):
        "Problem caused by your funds being low",
}


# Experiment: merge confusable classes (26 -> 19). Applied on top of `label`, never
# written to the processed files, so the original 26-class task stays intact.
_P = "Credit reporting | "
_D = "Debt collection | "
MERGE_MAP = {
    _P + "Incorrect information on your report": _P + "Dispute, accuracy or use of report",
    _P + "Problem with a credit reporting company's investigation into an existing problem":
        _P + "Dispute, accuracy or use of report",
    _P + "Improper use of your report": _P + "Dispute, accuracy or use of report",
    _D + "False statements or representation": _D + "Collector misconduct",
    _D + "Took or threatened to take negative or legal action": _D + "Collector misconduct",
    _D + "Communication tactics": _D + "Collector misconduct",
    "Mortgage | Trouble during payment process": "Mortgage | Payment trouble or hardship",
    "Mortgage | Struggling to pay mortgage": "Mortgage | Payment trouble or hardship",
    "Checking or savings account | Opening an account":
        "Checking or savings account | Opening or managing an account",
    "Checking or savings account | Managing an account":
        "Checking or savings account | Opening or managing an account",
    "Credit card | Other features, terms, or problems": "Credit card | Fees, interest or other terms",
    "Credit card | Fees or interest": "Credit card | Fees, interest or other terms",
}


def merge_labels(labels):
    """Map 26-class labels to the merged label set (unlisted labels stay as they are)."""
    return labels.replace(MERGE_MAP)


def load_raw(path):
    df=pd.read_parquet(path)
    df["date"]=pd.to_datetime(df["date"])
    return df

def clean(df):
    df = df.dropna(subset=["narrative"]).copy()
    df["narrative"] = df["narrative"].str.strip()
    df = df[df["narrative"].str.len() >= MIN_LEN]
    key = df["narrative"].str.lower().str.split().str.join(" ")
    df = df[~key.duplicated()]
    return df.reset_index(drop=True)

def add_label(df):
    df = df.copy()
    df["product"] = df["product"].replace(PRODUCT_MAP)
    df["issue"] = df["issue"].replace(ISSUE_MAP)
    for (product, old), new in PRODUCT_ISSUE_MAP.items():
        df.loc[(df["product"] == product) & (df["issue"] == old), "issue"] = new
    df["label"] = df["product"] + " | " + df["issue"]
    return df


N_CLASSES = 25


def split_by_date(df, train_frac=0.70, val_frac=0.15):
    """Oldest 70% -> train, next 15% -> val, newest 15% -> test."""
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    i = int(len(df) * train_frac)
    j = int(len(df) * (train_frac + val_frac))
    return df.iloc[:i].copy(), df.iloc[i:j].copy(), df.iloc[j:].copy()


def top_labels(train_df, n=N_CLASSES):
    """The n most common labels, chosen from the TRAIN split only (no leakage)."""
    return set(train_df["label"].value_counts().head(n).index)


def apply_other(df, keep):
    df = df.copy()
    df["label"] = df["label"].where(df["label"].isin(keep), "Other")
    return df


def build_splits(raw_path="data/raw/complaints.parquet", out_dir="data/processed"):
    from pathlib import Path

    df = add_label(clean(load_raw(raw_path)))
    train, val, test = split_by_date(df)
    keep = top_labels(train)
    splits = {"train": train, "val": val, "test": test}
    splits = {k: apply_other(v, keep) for k, v in splits.items()}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_parquet(Path(out_dir) / f"{name}.parquet", index=False)
    return splits


if __name__ == "__main__":
    for name, part in build_splits().items():
        print(name, len(part), part["date"].min().date(), part["date"].max().date(),
              part["label"].nunique())

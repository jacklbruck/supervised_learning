import pandas as pd

from category_encoders import TargetEncoder
from pathlib import Path


def get_dataset(dataset_id):
    # Set file location.
    fp = Path(__file__).parents[2] / "data" / dataset_id / "train.csv"

    # Open dataset.
    df = pd.read_csv(fp, index_col="id")

    # Unpack into features and labels.
    X, y = df[df.columns[df.columns != "target"]], df[["target"]]

    # Process data.
    X = TargetEncoder(cols=X.select_dtypes(object).columns).fit_transform(X, y)
    X = X.join(
        X.select_dtypes(float)
        .isna()
        .astype(int)
        .rename(lambda x: f"{x}:is_missing", axis=1)
    )
    X = X.fillna(0)

    return X, y

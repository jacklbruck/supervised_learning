import numpy as np

from pathlib import Path


def load_results(dataset_id, model_id):
    # Set file location.
    fp = (
        Path(__file__).parents[2]
        / "report"
        / "results"
        / dataset_id
        / model_id
        / "cv_results_.npy"
    )

    # Load results.
    cv_results_ = np.load(fp, allow_pickle=True)

    return cv_results_.item()

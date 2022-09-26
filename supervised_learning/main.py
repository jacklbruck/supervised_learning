import numpy as np
import logging

from .utils.model import get_grid_searcher
from .utils.data import get_dataset
from itertools import product
from pathlib import Path

def problems():
    dataset_ids = [
        fp.parent.stem
        for fp in Path(__file__).parents[1].rglob("data/*/*.csv")
    ]
    model_ids = [
        fp.stem for fp in Path(__file__).parent.rglob("config/*.yaml")
    ]

    return product(dataset_ids, model_ids)

def fit(dataset_id, model_id):
    # Load and preprocess dataset.
    X, y = get_dataset(dataset_id)

    # Create grid search object.
    clf = get_grid_searcher(model_id)

    # Run grid search.
    clf.fit(X, y)

    return clf


def write(clf, dataset_id, model_id):
    # Get output path.
    fp = (
        Path(__file__).parents[1]
        / "report"
        / "results"
        / dataset_id
        / model_id
    )
    fp.mkdir(exist_ok=True, parents=True)

    # Write output.
    np.save(fp / "cv_results_.npy", clf.cv_results_, allow_pickle=True)


def main():

    for dataset_id, model_id in problems():
        logging.info(f"Beginning {dataset_id}, {model_id}...")

        try:
            clf = fit(dataset_id, model_id)
        except Exception as e:
            logging.warn("Error in fit", e);

        try:
            write(clf, dataset_id, model_id)
        except Exception as e:
            logging.warn("Error in write", e);


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    main()

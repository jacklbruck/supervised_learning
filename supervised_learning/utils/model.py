from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.svm import SVC
from yaml import safe_load
from pathlib import Path


models = {
    "decision_trees": DecisionTreeClassifier,
    "neural_network": MLPClassifier,
    "boosting": GradientBoostingClassifier,
    "support_vector_machine": SVC,
    "k_nearest_neighbors": KNeighborsClassifier,
}


def get_param_grid(model_id):
    # Set file location.
    fp = Path(__file__).parents[1] / "config" / f"{model_id}.yaml"

    # Open config.
    with open(fp, "r") as f:
        config = safe_load(f)["params"]

    # Add classifier name to keys.
    for k, v in list(config.items()):
        config[f"clf__{k}"] = v

        del config[k]

    return config


def get_grid_searcher(model_id):
    # Open config file.
    param_grid = get_param_grid(model_id)


    # Get estimator.
    estimator = Pipeline([("smp", SMOTE()), ("clf", models[model_id]())])

    return GridSearchCV(
        estimator,
        param_grid,
        scoring=[
            "accuracy",
            "balanced_accuracy",
            "f1",
            "f1_weighted",
            "precision",
            "recall",
            "roc_auc",
        ],
        refit="f1_weighted",
        verbose=2,
        cv=3,
    )

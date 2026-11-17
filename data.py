# data.py
# Loads the dataset. Three choices (set DATASET below, or pass dataset="..." to load_data):
#   "breast_cancer" -> UCI Breast Cancer Wisconsin (Diagnostic), 569 patients, 30 features (default)
#   "sonar"         -> UCI Sonar (mines vs rocks), 208 samples, 60 features        (harder)
#   "ionosphere"    -> UCI Ionosphere (radar returns), 351 samples, 34 features    (harder)
# The breast cancer data is loaded the same way (sklearn's load_breast_cancer) as the reference
# GitHub project abhipatel35/svm-hyperparameter-optimization-for-breast-cancer.
# Sonar and ionosphere are downloaded once from OpenML and cached in the data/ folder.

import os
import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split

DATASET = "breast_cancer"      # "breast_cancer", "sonar" or "ionosphere"

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")       # downloaded datasets are cached here

# OpenML name + which original label becomes -1 (the "interesting" class, like malignant).
OPENML = {
    "sonar":      {"name": "sonar",      "negative": "Mine"},
    "ionosphere": {"name": "ionosphere", "negative": "b"},
}


def _load_openml(key):
    cache_file = os.path.join(DATA_DIR, key + ".npz")
    if os.path.exists(cache_file):
        d = np.load(cache_file, allow_pickle=False)
        return d["X"], d["y"]
    try:
        raw = fetch_openml(name=OPENML[key]["name"], version=1, as_frame=False, parser="auto")
    except Exception as err:
        raise RuntimeError(
            f"Could not download the '{key}' dataset from OpenML (are you offline?).\n"
            f"Connect to the internet once and run again; it will then be cached in:\n  {DATA_DIR}\n"
            f"Original error: {err}"
        ) from err
    X = raw.data.astype(float)
    y = np.where(raw.target == OPENML[key]["negative"], -1, 1)     # map labels to -1 / +1
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(cache_file, X=X, y=y)
    return X, y


def load_data(seed=42, dataset=None):
    """Returns X_train, X_test, y_train, y_test.

    X = the measurements of each sample (one row per sample).
    y = the label: -1 = the 'positive finding' class (malignant / mine / bad radar return),
        +1 = the other class (benign / rock / good radar return).
    """
    dataset = DATASET if dataset is None else dataset
    if dataset == "breast_cancer":
        d = load_breast_cancer()
        X = d.data            # shape (569, 30)
        y = 2 * d.target - 1  # sklearn: 0 = malignant, 1 = benign  ->  -1 = malignant, +1 = benign
    elif dataset in OPENML:
        X, y = _load_openml(dataset)
    else:
        raise ValueError("DATASET must be 'breast_cancer', 'sonar' or 'ionosphere'")

    # Keep 20% of the data aside as a final test set.
    # stratify=y keeps the same class ratio in both parts.
    # random_state=seed makes the split the same every time you run it.
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

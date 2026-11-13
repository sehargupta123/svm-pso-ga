# data.py
# Loads the Breast Cancer Wisconsin (Diagnostic) dataset.
# This is the SAME dataset used in the GitHub project
# abhipatel35/svm-hyperparameter-optimization-for-breast-cancer,
# loaded the same way (sklearn's load_breast_cancer) with the same 80/20 split.

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_data(seed=42):
    """Returns X_train, X_test, y_train, y_test.

    X = the 30 measurements of each tumour (one row per patient).
    y = the label: -1 = malignant (cancer), +1 = benign (not cancer).
    """
    dataset = load_breast_cancer()
    X = dataset.data      # shape (569, 30)
    y = dataset.target    # 0 = malignant, 1 = benign

    # An SVM needs the labels to be -1 and +1 instead of 0 and 1.
    y = 2 * y - 1         # 0 -> -1,  1 -> +1

    # Keep 20% of the data aside as a final test set.
    # stratify=y keeps the same cancer/no-cancer ratio in both parts.
    # random_state=seed makes the split the same every time you run it.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    return X_train, X_test, y_train, y_test

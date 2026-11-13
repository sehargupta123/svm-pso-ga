# STEP 3: Test our own SVM and compare it with scikit-learn's SVM.
# Run it with:  python step3_test_my_svm.py
# If the two columns of accuracy match closely, our SVM is correct.

import time
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data import load_data
from my_svm import MySVM

X_train, X_test, y_train, y_test = load_data()

# Scale the data (learn the scaling from the training set only).
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

settings = [(1, 0.01), (1, 0.1), (10, 0.01), (100, 0.001), (0.1, 0.1)]

print(f"{'C':>6} {'gamma':>7} | {'MySVM acc':>9} {'sklearn acc':>11} | {'MySVM SVs':>9} {'sklearn SVs':>11} | {'KKT ok %':>8} {'time (s)':>8}")
print("-" * 88)
for C, gamma in settings:
    start = time.time()
    mine = MySVM(C=C, gamma=gamma).fit(X_train_s, y_train)
    seconds = time.time() - start

    ref = SVC(C=C, gamma=gamma, kernel="rbf").fit(X_train_s, y_train)

    print(f"{C:>6} {gamma:>7} | {mine.score(X_test_s, y_test):>9.4f} {ref.score(X_test_s, y_test):>11.4f} | "
          f"{mine.n_support:>9} {ref.n_support_.sum():>11} | {mine.check_kkt():>8.1f} {seconds:>8.2f}")
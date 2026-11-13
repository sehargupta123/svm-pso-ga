# STEP 1: Check that Python and all the libraries are installed correctly.
# Run it with:  python step1_check_setup.py

import sys
print("Python version:", sys.version.split()[0])

import numpy
import sklearn
import cvxopt
import matplotlib
import pandas

print("numpy       ", numpy.__version__)
print("scikit-learn", sklearn.__version__)
print("cvxopt      ", cvxopt.__version__)
print("matplotlib  ", matplotlib.__version__)
print("pandas      ", pandas.__version__)
print("\nAll libraries are installed. You are ready for Step 2!")

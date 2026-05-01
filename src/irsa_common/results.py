"""Common result-directory conventions."""
import os

RESULTS_ROOT = os.path.join("results", "new")


def under_results(name):
    return os.path.join(RESULTS_ROOT, name)

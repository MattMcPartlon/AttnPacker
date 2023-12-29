"""Constants used in feature generation
"""
import numpy as np

rng = np.arange(1, 33).tolist()
DEFAULT_SEP_BINS = [-i for i in reversed(rng)] + [0] + rng # noqa
SMALL_SEP_BINS = [1e6, 30, 20, 15, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1]
SMALL_SEP_BINS = [-i for i in SMALL_SEP_BINS] + [0] + list(reversed(SMALL_SEP_BINS))
DEFAULT_PW_DIST_RADII = np.linspace(0, 20, 16).tolist()  # noqa

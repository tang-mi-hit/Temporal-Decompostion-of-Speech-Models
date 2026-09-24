import numpy as np

from speech_strf.evaluate import conditional_contribution, stable_proportion


def test_full_minus_reduced_and_guarded_proportion():
    assert np.isclose(conditional_contribution(0.4, 0.25), 0.15)
    assert np.isclose(stable_proportion(0.15, 0.4), 0.375)
    assert np.isnan(stable_proportion(0.1, 0.0))
    assert np.isnan(stable_proportion(0.1, -0.2))


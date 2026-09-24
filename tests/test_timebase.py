import numpy as np

from speech_strf.timebase import events_to_grid, make_time_grid, resample_continuous


def test_shared_grid_and_alignment():
    grid = make_time_grid(1.0, 50)
    assert len(grid.times) == 50
    assert grid.times[-1] == 0.98
    values = resample_continuous(
        np.array([[0.0], [1.0]]), np.array([0.0, 1.0]), grid.times
    )
    assert np.isclose(values[25, 0], 0.5)
    events = events_to_grid(np.array([0.1, 0.101]), 50, 50)
    assert events[5] == 2


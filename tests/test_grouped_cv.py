import numpy as np

from speech_strf.fit_encoding import grouped_splits


def test_grouped_splits_never_leak_story_frames():
    groups = np.repeat(["a", "b", "c", "d"], 10)
    for train, test in grouped_splits(groups, 4):
        assert set(groups[train]).isdisjoint(groups[test])
        assert len(set(groups[test])) == 1


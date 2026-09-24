from pathlib import Path

import yaml


def test_nonsemantic_controls_are_excluded_from_retained_pool():
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "data.yaml").read_text()
    )["data"]
    assert {"story17", "story18"}.isdisjoint(config["sample_ids"])
    assert config["excluded_ids"] == {
        "story17": "control_condition_nonsemantic_audio",
        "story18": "control_condition_nonsemantic_audio",
    }


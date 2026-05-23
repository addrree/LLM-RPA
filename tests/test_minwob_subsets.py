from app.browsergym_integration.miniwob_tasks import EXTRACTION_MINIWOB_TASK_NAMES, VISUAL_SPATIAL_MINIWOB_TASK_NAMES


def test_extraction_subset_excludes_find_midpoint():
    assert "find-midpoint" not in EXTRACTION_MINIWOB_TASK_NAMES


def test_visual_subset_includes_find_midpoint():
    assert "find-midpoint" in VISUAL_SPATIAL_MINIWOB_TASK_NAMES

"""Evaluation runner, metrics, and multi-profile run comparison."""
from src.evaluation.comparison import (
    auto_restore_drive_backups,
    compare_profiles,
    discover_profiles_by_mode,
    load_profile_results,
    run_three_way_comparison,
    run_two_way_comparison,
)
from src.evaluation.metrics import compute_appearance_coherence_metrics, iou
from src.evaluation.runner import (
    crop_original,
    preview,
    read_mask,
    read_rgb,
    resolve_sample,
    run_one,
    sha,
)

__all__ = [
    "auto_restore_drive_backups",
    "compare_profiles",
    "compute_appearance_coherence_metrics",
    "crop_original",
    "discover_profiles_by_mode",
    "iou",
    "load_profile_results",
    "preview",
    "read_mask",
    "read_rgb",
    "resolve_sample",
    "run_one",
    "run_three_way_comparison",
    "run_two_way_comparison",
    "sha",
]

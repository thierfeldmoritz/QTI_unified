"""Patho-focused QTI synthetic generation, inference, and plotting tools.

The package is intentionally small and standalone. Large generated data,
patient scans, covariance fits, and MLP checkpoints are referenced by explicit
paths supplied by CLI flags or environment variables.

Top-level exports are lazy so core math and generation modules can be imported
without loading optional plotting or checkpoint dependencies.
"""

__all__ = [
    "GeneratedPathoCase",
    "PathoCase",
    "PathoRunConfig",
    "available_scenarios",
    "compare_patho_predictions",
    "generate_patho_data",
    "generate_patho_suite",
    "resolve_data_root",
    "validate_gt",
]


def __getattr__(name: str) -> object:
    """Lazily resolve public convenience exports.

    Parameters
    ----------
    name
        Export name requested from the package root.

    Returns
    -------
    object
        Resolved function or class.
    """

    if name in {"GeneratedPathoCase", "PathoCase", "PathoRunConfig", "resolve_data_root"}:
        from . import config

        return getattr(config, name)
    if name in {"available_scenarios", "generate_patho_suite"}:
        from . import patho

        return getattr(patho, name)
    if name in {"compare_patho_predictions", "generate_patho_data", "validate_gt"}:
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module 'qti_unified' has no attribute {name!r}")

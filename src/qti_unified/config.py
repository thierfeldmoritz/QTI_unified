"""Configuration objects and path helpers for the patho workflow.

The module deliberately stores only small scalar configuration. Large files
such as generated NIfTI stacks, patient scans, covariance fits, and model
checkpoints are always referenced by external paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _package_root() -> Path:
    """Return the repository root when running from an editable checkout.

    Returns
    -------
    pathlib.Path
        The parent directory that contains ``src`` and the documentation files.
    """

    return Path(__file__).resolve().parents[2]


def resolve_data_root(data_root: str | Path | None = None) -> Path:
    """Resolve the generated-data root used by patho commands.

    Parameters
    ----------
    data_root
        Optional explicit data root. When absent, ``QTI_DATA_ROOT`` is used.
        If neither is set, ``<repo>/data`` is used.

    Returns
    -------
    pathlib.Path
        Absolute data root path. The path is not created by this helper.
    """

    raw = data_root or os.environ.get("QTI_DATA_ROOT")
    return Path(raw).expanduser().resolve() if raw else (_package_root() / "data").resolve()


def resolve_optional_path(value: str | Path | None, env_name: str) -> Path | None:
    """Resolve an optional external path from an argument or environment.

    Parameters
    ----------
    value
        Explicit path value supplied by the caller.
    env_name
        Environment variable to read when ``value`` is absent.

    Returns
    -------
    pathlib.Path | None
        Absolute resolved path, or ``None`` when no value is available.
    """

    raw = value or os.environ.get(env_name)
    return Path(raw).expanduser().resolve() if raw else None


@dataclass(frozen=True)
class PathoCase:
    """A generated DTD case before signal synthesis.

    Parameters
    ----------
    scenario
        Patho scenario name, used as the first output directory level.
    case
        Case tag, used as file stem and second output directory level.
    params
        List of diffusion tensor parameter dictionaries. Eigenvalues are in
        SI units (m^2/s), and orientation fields are a unit vector.
    metadata
        Small provenance dictionary; values must be JSON serializable.
    """

    scenario: str
    case: str
    params: list[dict[str, float]]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class GeneratedPathoCase:
    """A patho case after GT and optional signal generation.

    Parameters
    ----------
    case
        Source patho DTD case.
    gt
        Stored ground-truth metrics computed once from the DTD parameters.
    exact_signal
        Exact DTD mixture signal with shape ``(n_measurements,)`` or ``None``.
    cumexp_signal
        Optional cumulant-expansion signal with shape ``(n_measurements,)``.
    noisy_signals
        Optional noisy realizations with shape ``(n_realizations, n_measurements)``.
    paths
        Output paths written for this case.
    """

    case: PathoCase
    gt: dict[str, float | None]
    exact_signal: object | None = None
    cumexp_signal: object | None = None
    noisy_signals: object | None = None
    paths: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PathoRunConfig:
    """Runtime options shared by patho generation commands.

    Parameters
    ----------
    data_root
        Generated-data root. This directory is local and gitignored.
    xps_path
        Optional external XPS ``.mat`` file. When absent, a deterministic
        built-in 54-measurement protocol is used for smoke tests.
    snr
        Bulk SNR for noisy signal realizations. ``None`` disables noisy output.
    n_realizations
        Number of noisy signal files to write per case.
    n_tensors
        Default tensor count for generated patho cases.
    seed
        Base random seed. Per-case seeds are derived deterministically.
    s0
        Baseline signal amplitude used for exact and noisy signals.
    """

    data_root: Path
    xps_path: Path | None = None
    snr: float | None = 30.0
    n_realizations: int = 100
    n_tensors: int = 1500
    seed: int = 42
    s0: float = 1.0

"""Smoke test: the package imports and exposes a version. Keeps CI green
until real Phase-1 tests land with their features."""

import job_radar


def test_package_exposes_version() -> None:
    assert job_radar.__version__ == "0.1.0"

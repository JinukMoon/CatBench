"""CatBench version used at runtime (stamped into result files).

Must equal `version` in pyproject.toml (publish.yml checks the tag against pyproject;
tests/test_115.py checks the two agree)."""
__version__ = "1.1.5"

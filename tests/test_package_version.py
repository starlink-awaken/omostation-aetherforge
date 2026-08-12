from importlib.metadata import version

import aetherforge


def test_runtime_version_matches_package_metadata() -> None:
    assert aetherforge.__version__ == version("aetherforge")

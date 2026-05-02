import os
import subprocess
import sys
import maturin as _maturin


def _run_stub_gen():
    # The isolated build env sets PYTHONHOME to a venv that lacks the full stdlib.
    # Point to the base interpreter instead so the stub_gen binary can initialize Python.
    env = os.environ.copy()
    env["PYTHONHOME"] = sys.base_prefix
    subprocess.run(["cargo", "run", "--bin", "stub_gen"], check=True, env=env)


def get_requires_for_build_wheel(config_settings=None):
    return _maturin.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_editable(config_settings=None):
    return _maturin.get_requires_for_build_editable(config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    return _maturin.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    result = _maturin.build_wheel(wheel_directory, config_settings, metadata_directory)
    _run_stub_gen()
    return result


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    result = _maturin.build_editable(wheel_directory, config_settings, metadata_directory)
    _run_stub_gen()
    return result

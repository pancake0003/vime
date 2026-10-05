"""CPU-only tests for the CI GPU-lock helper's visibility normalization.

Regression coverage for the ROCm non-contiguous allocation bug: the pipeline pins
HIP_VISIBLE_DEVICES, but vime's device resolver reads CUDA_VISIBLE_DEVICES, so the two
must agree or a non-contiguous set (e.g. 4,5,6,7) resolves to the wrong local ordinal.
These run without GPUs — the lock files are plain flock files under a tmp dir.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

GPU_LOCK_EXEC = Path(__file__).resolve().parent / "ci" / "gpu_lock_exec.py"

CHILD = (
    "import os, json; "
    "print(json.dumps({'HIP': os.environ.get('HIP_VISIBLE_DEVICES'), "
    "'CUDA': os.environ.get('CUDA_VISIBLE_DEVICES')}))"
)


def _run(tmp_path, target_env_name):
    pattern = str(tmp_path / "lock_{gpu_id}.lock")
    out = subprocess.check_output(
        [
            sys.executable,
            str(GPU_LOCK_EXEC),
            "--count",
            "1",
            "--total-gpus",
            "2",
            "--target-env-name",
            target_env_name,
            "--lock-path-pattern",
            pattern,
            "--",
            sys.executable,
            "-c",
            CHILD,
        ],
        text=True,
    )
    # The helper prints "[gpu_lock_exec] Acquired GPUs: ..." before the child output.
    line = [ln for ln in out.splitlines() if ln.strip().startswith("{")][-1]
    return json.loads(line)


@pytest.mark.unit
def test_hip_target_also_normalizes_cuda_visibility(tmp_path):
    """ROCm path: pinning HIP must also set CUDA_VISIBLE_DEVICES to the same set."""
    env = _run(tmp_path, "HIP_VISIBLE_DEVICES")
    assert env["HIP"] is not None
    assert env["CUDA"] == env["HIP"], "CUDA visibility must match the acquired HIP set"


@pytest.mark.unit
def test_cuda_target_unchanged(tmp_path):
    """CUDA path is a no-op change: the child sees CUDA set, HIP untouched (unset)."""
    env = _run(tmp_path, "CUDA_VISIBLE_DEVICES")
    assert env["CUDA"] is not None
    assert env["HIP"] is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

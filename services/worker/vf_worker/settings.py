"""Private, operator-controlled execution settings; never accepted from job JSON."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from vf_api.settings import EnvironmentSettings


class WorkerSettings(EnvironmentSettings):
    worker_ml_python: Path
    worker_work_dir: Path
    worker_cache_dir: Path
    worker_gpu_id: int = Field(ge=0)
    worker_device: Literal["cuda", "cpu"] = "cuda"
    worker_cpu_threads: int = Field(default=2, ge=1, le=16)
    worker_timeout_seconds: int = Field(default=86400, ge=30, le=604800)

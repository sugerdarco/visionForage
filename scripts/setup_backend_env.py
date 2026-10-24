"""Prepare shared local directories without replacing existing service secrets."""

import os
from pathlib import Path


def configure(root):
    path = root / ".env"
    if not path.exists():
        raise ValueError("Run make env before configuring the backend")
    values = {
        line.split("=", 1)[0].strip(): line.split("=", 1)[1].strip().strip("\"'")
        for line in path.read_text().splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    }
    artifacts = Path(values.get("ARTIFACTS_DIR") or root / "artifacts").absolute()
    if any(
        character in str(artifacts) for character in ("\n", "\r", ":", ",", "$", "'")
    ):
        raise ValueError("Unsupported characters in artifact directory")
    additions = {
        "ARTIFACTS_DIR": str(artifacts),
        "LOCAL_UID": str(os.getuid()),
        "LOCAL_GID": str(os.getgid()),
        "DOCKER_GID": "0",
    }
    # Native processes and containers must resolve the same host paths.
    for name, subdir in (
        ("DEPLOYMENT_WORK_DIR", "deployments"),
        ("PREDICTION_LOG_DIR", "predictions"),
    ):
        configured = Path(values.get(name) or artifacts / subdir)
        configured = configured if configured.is_absolute() else root / configured
        if configured.absolute() != artifacts / subdir:
            raise ValueError(f"{name} must resolve inside ARTIFACTS_DIR for Compose")
    for subdir in ("deployments", "predictions"):
        target = artifacts / subdir
        target.mkdir(parents=True, exist_ok=True, mode=0o700)
        if target.is_symlink():
            raise ValueError("Backend directories must not be symlinks")
        target.chmod(0o700)
    with path.open("a") as output:
        os.fchmod(output.fileno(), 0o600)
        for key, value in additions.items():
            if not values.get(key):
                output.write(f"\n{key}='{value}'\n")


if __name__ == "__main__":
    configure(Path(__file__).resolve().parents[1])
    print("Backend paths and local identity configured; service credentials preserved.")

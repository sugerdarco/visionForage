"""Prepare verified bundles and operate deterministic, owner-labeled containers."""

from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time

from vf_api.deployment_runtime import DeploymentUnavailable, DockerRuntime


class DeploymentManager:
    def __init__(self, settings, runtime=None):
        self.settings = settings
        self.runtime = runtime or DockerRuntime(settings)
        self.root = Path(settings.deployment_work_dir).absolute()

    def prepare(self, spec, store):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.root.is_symlink() or self.root.stat().st_mode & 0o077:
            raise DeploymentUnavailable("Deployment directory must be private")
        target = self.root / str(spec["id"])
        if target.exists() or target.is_symlink():
            raise DeploymentUnavailable("Deployment bundle already exists")
        with TemporaryDirectory(dir=self.root, prefix=".preparing-") as temporary:
            source = Path(temporary)
            for name, item in (
                ("model.onnx", spec["model"]),
                ("metadata.json", spec["metadata"]),
            ):
                with (source / name).open("xb") as output:
                    store.download(item["key"], output, item["bytes"], item["sha256"])
                (source / name).chmod(0o444)
            # The private parent prevents host users reading this directory; the
            # bind-mounted directory itself must be readable by the container UID.
            source.chmod(0o755)
            source.rename(target)
        return target

    def cleanup(self, spec):
        self.runtime.remove(spec)
        target = self.root / str(spec["id"])
        if self.root.is_symlink() or target.is_symlink():
            raise DeploymentUnavailable("Invalid deployment directory")
        try:
            if target.exists():
                shutil.rmtree(target)
        except OSError:
            raise DeploymentUnavailable("Deployment cleanup unavailable") from None

    def launch(self, spec, store, image, check):
        bundle = self.prepare(spec, store)
        root = self.settings.prediction_log_dir.absolute()
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if root.is_symlink() or root.stat().st_mode & 0o077:
            raise DeploymentUnavailable("Prediction directory must be private")
        outbox = root / str(spec["id"])
        outbox.mkdir(mode=0o700)
        # Private parent protects host access; container UID can write its mount.
        outbox.chmod(0o1777)
        spec = {**spec, "outbox": outbox}
        check()
        self.runtime.create(spec, bundle, image)
        check()
        self.runtime.start(spec)
        deadline = time.monotonic() + self.settings.deployment_ready_seconds
        while time.monotonic() < deadline:
            check()
            info = self.runtime.inspect(spec)
            if self.runtime.healthy(info) and self.runtime.route_ready(info, spec):
                return info
            if info is None or not info.get("State", {}).get("Running"):
                raise DeploymentUnavailable("Inference startup failed")
            if info["State"].get("Health", {}).get("Status") == "unhealthy":
                raise DeploymentUnavailable("Inference startup failed")
            time.sleep(0.5)
        raise DeploymentUnavailable("Inference readiness timed out")

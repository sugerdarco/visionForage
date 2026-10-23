"""Bounded Docker CLI operations, restricted to this app's labeled containers."""

import json
import subprocess


class DeploymentUnavailable(RuntimeError):
    pass


class DockerRuntime:
    def __init__(self, settings):
        self.settings = settings

    def command(self, *args, input=None):
        try:
            result = subprocess.run(
                ["docker", *args],
                input=input,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            raise DeploymentUnavailable("Container runtime unavailable") from None
        if result.returncode or len(result.stdout) > 1024**2:
            raise DeploymentUnavailable("Container operation failed")
        return result.stdout.strip()

    def name(self, identifier):
        return f"{self.settings.deployment_namespace}-infer-{identifier.hex}"

    def labels(self, spec):
        return {
            "io.visionforge.scope": self.settings.deployment_namespace,
            "io.visionforge.deployment": str(spec["id"]),
            "io.visionforge.model": str(spec["model_id"]),
            "io.visionforge.project": str(spec["project_id"]),
        }

    def image(self):
        return self.command(
            "image", "inspect", "--format", "{{.Id}}", self.settings.deployment_image
        )

    def inspect(self, spec):
        # A successful list distinguishes absence from an unreachable daemon.
        found = self.command(
            "container",
            "ls",
            "--all",
            "--filter",
            f"name=^/{self.name(spec['id'])}$",
            "--format",
            "{{.ID}}",
        )
        if not found:
            return None
        try:
            info = json.loads(
                self.command("container", "inspect", self.name(spec["id"]))
            )[0]
            labels = info["Config"].get("Labels") or {}
            if any(labels.get(k) != v for k, v in self.labels(spec).items()):
                raise DeploymentUnavailable("Container ownership mismatch")
            return info
        except (ValueError, KeyError, IndexError, TypeError):
            raise DeploymentUnavailable("Invalid container response") from None

    def create(self, spec, bundle, image):
        env = {
            "INFERENCE_BUNDLE_DIR": "/bundle",
            "INFERENCE_MODEL_ID": str(spec["model_id"]),
            "INFERENCE_MODEL_VERSION": str(spec["model_version"]),
            "INFERENCE_SLUG": spec["slug"],
            "INFERENCE_MODEL_SHA256": spec["model"]["sha256"],
            "INFERENCE_METADATA_SHA256": spec["metadata"]["sha256"],
            "INFERENCE_API_KEY_SHA256": spec["key_hash"],
            "INFERENCE_CPU_THREADS": str(self.settings.deployment_cpu_threads),
        }
        labels = [
            argument
            for k, v in self.labels(spec).items()
            for argument in ("--label", f"{k}={v}")
        ]
        self.command(
            "container",
            "create",
            "--name",
            self.name(spec["id"]),
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "128",
            "--memory",
            f"{self.settings.deployment_memory_mb}m",
            "--memory-swap",
            f"{self.settings.deployment_memory_mb}m",
            "--cpus",
            str(self.settings.deployment_cpu_threads),
            "--tmpfs",
            "/tmp:rw,nosuid,noexec,size=64m",
            "--log-driver",
            "json-file",
            "--log-opt",
            "max-size=10m",
            "--log-opt",
            "max-file=2",
            "--publish",
            "127.0.0.1::8000",
            "--mount",
            f"type=bind,src={bundle},dst=/bundle,readonly",
            "--env-file",
            "/dev/stdin",
            *labels,
            image,
            input="".join(f"{k}={v}\n" for k, v in env.items()),
        )

    def start(self, spec):
        info = self.inspect(spec)
        if info is None:
            raise DeploymentUnavailable("Container disappeared")
        self.command("container", "start", info["Id"])

    def remove(self, spec):
        info = self.inspect(spec)
        if info is not None:
            self.command("container", "rm", "--force", info["Id"])
        if self.inspect(spec) is not None:
            raise DeploymentUnavailable("Container removal not confirmed")

    @staticmethod
    def endpoint(info, slug):
        try:
            bindings = info["NetworkSettings"]["Ports"]["8000/tcp"]
            if len(bindings) != 1 or bindings[0]["HostIp"] != "127.0.0.1":
                raise ValueError()
            port = int(bindings[0]["HostPort"])
            if not 1 <= port <= 65535:
                raise ValueError()
            return f"http://127.0.0.1:{port}/predict/{slug}"
        except (ValueError, KeyError, IndexError, TypeError):
            raise DeploymentUnavailable("Invalid endpoint binding") from None

    @staticmethod
    def healthy(info):
        return bool(
            info
            and info.get("State", {}).get("Running") is True
            and info["State"].get("Health", {}).get("Status") == "healthy"
        )

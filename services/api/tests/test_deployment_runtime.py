"""Check the ownership boundary before issuing destructive Docker commands."""

import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from uuid import uuid4

from vf_api.deployment_runtime import DeploymentUnavailable, DockerRuntime
from vf_api.settings import DeploymentSettings


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        self.runtime = DockerRuntime(DeploymentSettings())
        self.spec = {name: uuid4() for name in ("id", "model_id", "project_id")}

    def test_foreign_labels_prevent_start_and_removal(self):
        info = {"Id": "foreign", "Config": {"Labels": self.runtime.labels(self.spec)}}
        info["Config"]["Labels"]["io.visionforge.project"] = str(uuid4())
        for operation in (self.runtime.start, self.runtime.remove):
            with patch.object(
                self.runtime, "command", side_effect=["foreign", json.dumps([info])]
            ) as command:
                with self.assertRaises(DeploymentUnavailable):
                    operation(self.spec)
                self.assertEqual(command.call_count, 2)
                self.assertFalse(
                    any(
                        "rm" in c.args or "start" in c.args
                        for c in command.call_args_list
                    )
                )

    def test_removal_uses_verified_id_and_confirms_absence(self):
        info = {"Id": "owned-id", "Config": {"Labels": self.runtime.labels(self.spec)}}
        with patch.object(
            self.runtime,
            "command",
            side_effect=["owned-id", json.dumps([info]), "owned-id", ""],
        ) as command:
            self.runtime.remove(self.spec)
        self.assertEqual(
            command.call_args_list[2].args, ("container", "rm", "--force", "owned-id")
        )
        self.assertEqual(command.call_count, 4)

    def test_runtime_errors_do_not_expose_docker_output(self):
        for result in (
            subprocess.CompletedProcess([], 1, "private stdout", "private stderr"),
            subprocess.TimeoutExpired([], 30, output="private stdout"),
        ):
            kwargs = (
                {"side_effect": result}
                if isinstance(result, Exception)
                else {"return_value": result}
            )
            with patch("vf_api.deployment_runtime.subprocess.run", **kwargs):
                with self.assertRaises(DeploymentUnavailable) as error:
                    self.runtime.command("info")
                self.assertNotIn("private", str(error.exception))

    def test_non_loopback_bindings_are_rejected(self):
        for bindings in (
            None,
            [],
            [{"HostIp": "0.0.0.0", "HostPort": "8000"}],
            [{"HostIp": "127.0.0.1", "HostPort": "0"}],
        ):
            with self.assertRaises(DeploymentUnavailable):
                self.runtime.endpoint(
                    {"NetworkSettings": {"Ports": {"8000/tcp": bindings}}}, "test"
                )

    def test_create_limits_resources_and_passes_only_serving_configuration(self):
        spec = {
            **self.spec,
            "model_version": 1,
            "slug": "test",
            "outbox": Path("/private/outbox"),
            "key_hash": "a" * 64,
            "model": {"sha256": "b" * 64},
            "metadata": {"sha256": "c" * 64},
        }
        with patch.object(self.runtime, "command") as command:
            self.runtime.create(spec, Path("/private/bundle"), "sha256:image")
        args = command.call_args.args
        self.assertEqual(args[-1], "sha256:image")
        for flag, value in (
            ("--cap-drop", "ALL"),
            ("--network", "visionforge_inference"),
            ("--memory", "1536m"),
            ("--memory-swap", "1536m"),
            ("--cpus", "2"),
            ("--security-opt", "no-new-privileges"),
        ):
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertIn("--read-only", args)
        self.assertNotIn("--publish", args)
        self.assertIn("traefik.enable=true", args)
        self.assertIn(
            f"traefik.http.routers.{self.runtime.name(spec['id'])}.rule=Path(`/predict/test`)",
            args,
        )
        self.assertEqual(
            args[args.index("--mount") + 1],
            "type=bind,src=/private/bundle,dst=/bundle,readonly",
        )
        env = command.call_args.kwargs["input"].splitlines()
        self.assertEqual(len(env), 10)
        self.assertTrue(all(line.startswith("INFERENCE_") for line in env))

    def test_route_probe_requires_expected_deployment_and_auth_rejection(self):
        spec = {**self.spec, "slug": "test"}
        info = {
            "Config": {
                "Labels": {
                    "traefik.enable": "true",
                    "traefik.docker.network": "serving",
                    "io.visionforge.endpoint": "http://127.0.0.1:8080/predict/test",
                }
            },
            "NetworkSettings": {"Networks": {"serving": {}}},
        }
        self.assertEqual(
            self.runtime.endpoint(info, "test"), "http://127.0.0.1:8080/predict/test"
        )
        with self.assertRaises(DeploymentUnavailable):
            self.runtime.endpoint(info, "another-model")
        for code, identifier, expected in (
            (401, str(spec["id"]), True),
            (404, str(spec["id"]), False),
            (401, str(uuid4()), False),
        ):
            with patch("vf_api.deployment_runtime.build_opener") as opener:
                opener.return_value.open.side_effect = HTTPError(
                    "http://test",
                    code,
                    "probe",
                    {"X-VisionForge-Deployment": identifier},
                    None,
                )
                self.assertEqual(self.runtime.route_ready(info, spec), expected)

"""Check setup guardrails without installing dependencies or touching a GPU."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


class EnvironmentScriptsTest(unittest.TestCase):
    def test_setup_requires_absolute_directory(self) -> None:
        for arguments in ([], ["relative-directory"]):
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    ["bash", str(REPO / "scripts/setup_training_env.sh"), *arguments],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("Usage:", result.stderr)

    def test_invalid_backend_does_not_create_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "runtime"
            result = subprocess.run(
                ["bash", str(REPO / "scripts/setup_training_env.sh"), str(target), "typo"],
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertFalse(target.exists())

    def test_cuda_requires_explicit_gpu_selection(self) -> None:
        environment = dict(os.environ)
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts/check_torch.py"), "--device", "cuda"],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("CUDA_VISIBLE_DEVICES", result.stderr)


if __name__ == "__main__":
    unittest.main()

"""Keep generated local credentials private and preserve existing configuration."""

from pathlib import Path
import tempfile
import unittest

from scripts.setup_local_env import create_env


class LocalEnvTest(unittest.TestCase):
    def test_random_secrets_permissions_and_no_overwrite(self):
        template = (Path(__file__).resolve().parents[1] / ".env.example").read_text()
        with tempfile.TemporaryDirectory() as work:
            first, second = Path(work) / "first", Path(work) / "second"
            create_env(first, template)
            create_env(second, template)
            original = first.read_text()
            values = dict(line.split("=", 1) for line in original.splitlines() if "=" in line)
            names = ("POSTGRES_PASSWORD", "REDIS_PASSWORD", "MINIO_ROOT_PASSWORD")
            self.assertEqual(len({values[name] for name in names}), 3)
            for name in names:
                self.assertRegex(values[name], r"^[a-f0-9]{48}$")
                self.assertNotIn(values[name], second.read_text())
            self.assertEqual(first.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                create_env(first, template)
            self.assertEqual(first.read_text(), original)

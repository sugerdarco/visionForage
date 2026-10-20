"""Database configuration must not expose passwords or import training packages."""

import subprocess
import sys
import unittest

from pydantic import ValidationError
from sqlalchemy.engine import make_url

from vf_api.settings import Settings


class SettingsTest(unittest.TestCase):
    def test_password_special_characters_and_redaction(self):
        password = "example:@/% password"
        settings = Settings(_env_file=None, postgres_password=password)
        self.assertNotIn(password, repr(settings))
        self.assertNotIn(password, str(settings.database_url))
        rendered = settings.database_url.render_as_string(hide_password=False)
        self.assertEqual(make_url(rendered).password, password)

    def test_invalid_port_and_empty_password(self):
        for values in (
            {"postgres_password": ""},
            {"postgres_password": "example", "postgres_port": 0},
        ):
            with self.assertRaises(ValidationError):
                Settings(_env_file=None, **values)

    def test_imports_do_not_load_ml_runtime(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import vf_api.models; import vf_api.database; "
                "assert not {'torch', 'timm', 'ultralytics'} & set(sys.modules)",
            ],
            check=True,
        )

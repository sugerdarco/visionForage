"""Create local credentials; explicitly add an API key to older environments."""

import argparse
import os
from pathlib import Path
import secrets


def create_env(destination: Path, template: str) -> None:
    for name in ("POSTGRES_PASSWORD", "REDIS_PASSWORD", "MINIO_ROOT_PASSWORD"):
        template = template.replace(f"{name}=\n", f"{name}={secrets.token_hex(24)}\n")
    template = template.replace(
        "JWT_SECRET=\n", f"JWT_SECRET={secrets.token_hex(32)}\n"
    )
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as file:
        file.write(template)


def add_api_secret(destination: Path) -> bool:
    """Append a missing key without rewriting or rotating existing credentials."""
    with destination.open("r+") as file:
        original = file.read()
        for line in original.splitlines():
            setting = line.strip().removeprefix("export ")
            if setting.split("=", 1)[0].strip() == "JWT_SECRET":
                return False
        os.fchmod(file.fileno(), 0o600)
        separator = "" if not original or original.endswith("\n") else "\n"
        file.write(f"{separator}JWT_SECRET={secrets.token_hex(32)}\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--add-api-secret",
        action="store_true",
        help="Append JWT_SECRET to an existing .env only when the setting is absent.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        create_env(root / ".env", (root / ".env.example").read_text())
    except FileExistsError:
        if args.add_api_secret and add_api_secret(root / ".env"):
            print("Added a random API signing key; existing credentials are unchanged.")
        else:
            print("Keeping the existing .env file.")
    else:
        print("Created .env with random local passwords and owner-only permissions.")

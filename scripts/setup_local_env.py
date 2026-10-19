"""Create local credentials without replacing an existing .env file."""

import os
from pathlib import Path
import secrets


def create_env(destination: Path, template: str) -> None:
    for name in ("POSTGRES_PASSWORD", "REDIS_PASSWORD", "MINIO_ROOT_PASSWORD"):
        template = template.replace(f"{name}=\n", f"{name}={secrets.token_hex(24)}\n")
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as file:
        file.write(template)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    try:
        create_env(root / ".env", (root / ".env.example").read_text())
    except FileExistsError:
        print("Keeping the existing .env file.")
    else:
        print("Created .env with random local passwords and owner-only permissions.")

"""Small S3 boundary shared by the API and the CPU worker."""

import base64
import hashlib
from typing import BinaryIO

import boto3
from botocore.config import Config

from vf_api.settings import StorageSettings

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
UPLOAD_SECONDS = 900
READ_SECONDS = 300


class ObjectStore:
    def __init__(self, settings: StorageSettings):
        self.bucket = settings.s3_images_bucket
        options = {
            "aws_access_key_id": settings.minio_root_user,
            "aws_secret_access_key": settings.minio_root_password.get_secret_value(),
            "region_name": settings.s3_region,
            "config": Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
                connect_timeout=3,
                read_timeout=15,
                retries={"max_attempts": 2, "mode": "standard"},
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
            ),
        }
        sdk = boto3.session.Session()
        self.client = sdk.client("s3", endpoint_url=settings.endpoint, **options)
        self.public = sdk.client("s3", endpoint_url=settings.public_endpoint, **options)

    def close(self) -> None:
        self.client.close()
        self.public.close()

    def presign_put(
        self, key: str, mime: str, size: int, sha256: str
    ) -> tuple[str, dict]:
        checksum = base64.b64encode(bytes.fromhex(sha256)).decode("ascii")
        url = self.public.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": self.bucket,
                "Key": key,
                "ContentType": mime,
                "ContentLength": size,
                "ChecksumSHA256": checksum,
            },
            ExpiresIn=UPLOAD_SECONDS,
        )
        return url, {"Content-Type": mime, "x-amz-checksum-sha256": checksum}

    def read(self, key: str, *, max_bytes: int = MAX_UPLOAD_BYTES) -> bytes:
        result = self.client.get_object(Bucket=self.bucket, Key=key)
        stream = result["Body"]
        try:
            if result["ContentLength"] > max_bytes:
                raise ValueError("Object exceeds the configured read limit")
            data = stream.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise ValueError("Object exceeds the configured read limit")
            return data
        finally:
            stream.close()

    def put(self, key: str, data: bytes, mime: str) -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=mime,
        )

    def download(self, key: str, target: BinaryIO, size: int, sha256: str) -> None:
        """Stream an archive into a private temporary file; verify the signed claim."""
        result = self.client.get_object(Bucket=self.bucket, Key=key)
        stream = result["Body"]
        try:
            if result["ContentLength"] != size:
                raise ValueError("Uploaded archive size does not match the grant")
            digest = hashlib.sha256()
            received = 0
            while chunk := stream.read(min(1024 * 1024, size - received + 1)):
                received += len(chunk)
                if received > size:
                    raise ValueError("Uploaded archive exceeds the signed size")
                target.write(chunk)
                digest.update(chunk)
            if received != size or digest.hexdigest() != sha256:
                raise ValueError(
                    "Uploaded archive size or SHA-256 does not match the grant"
                )
            target.seek(0)
        finally:
            stream.close()

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def url(self, key: str) -> str:
        return self.public.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=READ_SECONDS,
        )

    def configure_staging_expiry(self) -> None:
        """Preserve other lifecycle rules while expiring temporary uploads."""
        from botocore.exceptions import ClientError

        try:
            rules = self.client.get_bucket_lifecycle_configuration(Bucket=self.bucket)[
                "Rules"
            ]
        except ClientError as error:
            if error.response["Error"]["Code"] != "NoSuchLifecycleConfiguration":
                raise
            rules = []
        rule_id = "visionforge-expire-upload-staging"
        rules = [rule for rule in rules if rule.get("ID") != rule_id]
        rules.append(
            {
                "ID": rule_id,
                "Status": "Enabled",
                "Filter": {"Prefix": "uploads/"},
                "Expiration": {"Days": 1},
            }
        )
        self.client.put_bucket_lifecycle_configuration(
            Bucket=self.bucket,
            LifecycleConfiguration={"Rules": rules},
        )


class DatasetStore(ObjectStore):
    """Separate private bucket and create-only writes for frozen JSON artifacts."""

    def __init__(self, settings: StorageSettings):
        super().__init__(settings)
        self.bucket = settings.s3_datasets_bucket

    def put_once(self, key: str, data: bytes) -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType="application/json",
            IfNoneMatch="*",
            ChecksumSHA256=base64.b64encode(hashlib.sha256(data).digest()).decode(
                "ascii"
            ),
        )


if __name__ == "__main__":
    ObjectStore(StorageSettings()).configure_staging_expiry()
    print(
        "Temporary upload objects expire after one day; registered images are unaffected."
    )

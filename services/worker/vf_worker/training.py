"""Claim one job, supervise isolated ML execution, then publish a verified model."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
from tempfile import TemporaryDirectory
import time
import traceback
from uuid import UUID, uuid4

from pydantic import TypeAdapter
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert

from vf_api.job_schemas import JobCreate
from vf_api.models import DatasetVersion, Model, Project, TrainingJob, TrainingMetric
from vf_worker.frozen_data import materialize


class Cancelled(Exception):
    pass


class WorkerBusy(Exception):
    pass


@contextmanager
def gpu_slot(settings):
    root = settings.worker_work_dir.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / f"gpu-{settings.worker_gpu_id}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise WorkerBusy() from None
        try:
            yield root
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def claim(sessions, identifier, task_id, gpu_id):
    with sessions.begin() as session:
        job = session.scalar(
            select(TrainingJob).where(TrainingJob.id == identifier).with_for_update()
        )
        if (
            job is None
            or job.status != "queued"
            or job.cancel_requested_at is not None
            or job.celery_task_id != task_id
        ):
            return None
        job.status, job.gpu_id = "running", gpu_id
        job.started_at = session.scalar(select(func.clock_timestamp()))
        version = session.get(DatasetVersion, job.dataset_version_id)
        task = {"resnet18": "classification", "yolov8s": "detection"}.get(
            job.model_arch
        )
        if task is None:
            raise ValueError("Unsupported model architecture")
    return job, version, task


def check_active(sessions, identifier):
    with sessions() as session:
        row = session.execute(
            select(TrainingJob.status, TrainingJob.cancel_requested_at).where(
                TrainingJob.id == identifier
            )
        ).one()
        if row.status != "running" or row.cancel_requested_at is not None:
            raise Cancelled()


def record_history(sessions, identifier, path, after):
    if not path.exists():
        return after
    if path.stat().st_size > 32 * 1024**2:
        raise ValueError("Training history exceeds limit")
    rows = json.loads(path.read_text())
    values = []
    for row in rows:
        epoch = row["epoch"]
        if type(epoch) is not int or not 1 <= epoch <= 1000:
            raise ValueError("Invalid training epoch")
        if epoch <= after:
            continue
        if "metrics" in row:
            metrics = {("val", "fitness"): row["fitness"]}
            for key, value in row["metrics"].items():
                prefix, name = key.split("/", 1)
                if prefix not in ("train", "val", "metrics", "lr"):
                    raise ValueError("Unknown detector metric prefix")
                metrics[
                    (
                        "train" if prefix in ("train", "lr") else "val",
                        key if prefix == "lr" else name,
                    )
                ] = value
        else:
            metrics = {
                (split, name): row[split][name]
                for split in ("train", "val")
                for name in ("loss", "accuracy")
            }
            metrics[("train", "learning_rate")] = row["learning_rate"]
        for (split, name), value in metrics.items():
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or len(name) > 100
            ):
                raise ValueError("Invalid epoch metric")
            values.append(
                dict(
                    job_id=identifier,
                    epoch=epoch,
                    split=split,
                    name=name,
                    value=float(value),
                )
            )
    if values:
        with sessions.begin() as session:
            job = session.scalar(
                select(TrainingJob)
                .where(TrainingJob.id == identifier)
                .with_for_update()
            )
            if job.status != "running" or job.cancel_requested_at is not None:
                raise Cancelled()
            session.execute(
                insert(TrainingMetric)
                .values(values)
                .on_conflict_do_nothing(constraint="uq_training_metrics_epoch_value")
            )
    return max([after] + [row["epoch"] for row in rows])


def stop_process(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def execute(spec, settings, log_path, check, progress):
    work = Path(spec["work"])
    spec_path = work / "request.json"
    spec_path.write_text(json.dumps(spec, allow_nan=False))
    repository = Path(__file__).resolve().parents[3]
    environment = {
        key: value
        for key, value in os.environ.items()
        if not any(
            word in key.upper() for word in ("PASSWORD", "SECRET", "TOKEN", "AWS_")
        )
    }
    environment.update(
        PYTHONPATH=os.pathsep.join(
            str(repository / path) for path in ("packages", "services/worker")
        ),
        CUDA_VISIBLE_DEVICES=str(settings.worker_gpu_id)
        if settings.worker_device == "cuda"
        else "",
        OMP_NUM_THREADS=str(settings.worker_cpu_threads),
        MKL_NUM_THREADS=str(settings.worker_cpu_threads),
        PYTHONUNBUFFERED="1",
    )
    environment.pop("PYTHONHOME", None)
    check()
    with log_path.open("wb") as log:
        log_path.chmod(0o600)
        process = subprocess.Popen(
            [
                # Preserve the venv symlink: resolving it bypasses its site-packages.
                str(settings.worker_ml_python.absolute()),
                "-m",
                "vf_worker.pipeline",
                str(spec_path),
                str(os.getpid()),
            ],
            cwd=work,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + settings.worker_timeout_seconds
        try:
            while True:
                check()
                progress()
                if process.poll() is not None:
                    break
                if (
                    time.monotonic() >= deadline
                    or log_path.stat().st_size > 16 * 1024**2
                ):
                    raise RuntimeError("Training process exceeded time or log limit")
                time.sleep(1)
            if process.returncode != 0:
                raise RuntimeError("Training subprocess failed")
            progress()
        finally:
            stop_process(process)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_report(path):
    if path.stat().st_size > 32 * 1024**2:
        raise ValueError("Training report exceeds limit")

    def reject_constant(value):
        raise ValueError("Nonfinite report")

    value = json.loads(path.read_text(), parse_constant=reject_constant)
    # JSON exponents such as 1e999 can overflow without invoking parse_constant.
    json.dumps(value, allow_nan=False)
    return value


def verify_bundle(work, manifest, manifest_sha):
    metadata = read_report(work / "onnx/metadata.json")
    verification = read_report(work / "onnx/verification.json")
    evaluation = read_report(work / "evaluation.json")
    weights_sha, onnx_sha = (
        digest(work / "run/best.pt"),
        digest(work / "onnx/model.onnx"),
    )
    for report in (metadata, evaluation):
        if (
            report["manifest_sha256"] != manifest_sha
            or report["checkpoint_sha256"] != weights_sha
            or report["task_type"] != manifest.task_type
        ):
            raise ValueError("Training artifact provenance mismatch")
    parity = verification["parity"]
    if (
        metadata["model_sha256"] != onnx_sha
        or verification["model_sha256"] != onnx_sha
        or metadata["class_map"] != dict(manifest.class_map)
        or metadata["opset"] != 17
        or evaluation["split"] != "test"
        or evaluation.get("smoke_run", False)
        or parity["passed"] is not True
        or parity["meets_32_test_image_check"] is not True
        or parity["split"] != "test"
        or parity["samples"] != 32
        or parity["batch_sizes"] != [1, 8]
        or parity["relative_tolerance"] != 0
        or len(set(parity["image_keys"])) != 32
        or not set(parity["image_keys"])
        <= {sample.image_key for sample in manifest.samples_for("test")}
    ):
        raise ValueError("Training export did not pass the required gates")
    limits = (
        [("max_absolute_error", "absolute_tolerance", 0.001)]
        if manifest.task_type == "classification"
        else [
            ("max_box_absolute_error", "box_absolute_tolerance", 0.002),
            ("max_score_absolute_error", "score_absolute_tolerance", 0.001),
        ]
    )
    for field, tolerance, maximum in limits:
        if parity[tolerance] != maximum or not 0 <= parity[field] <= maximum:
            raise ValueError("ONNX tolerance gate failed")
    shape = metadata["input"]["shape"]
    if (
        len(shape) != 4
        or shape[:2] != ["batch", 3]
        or type(shape[2]) is not int
        or shape[2] != shape[3]
        or shape[2] < 1
    ):
        raise ValueError("Invalid model input shape")
    return evaluation["metrics"], verification, shape[2]


def register(
    sessions, identifier, project_id, class_map, metrics, artifacts, input_size
):
    with sessions.begin() as session:
        session.execute(
            select(Project.id).where(Project.id == project_id).with_for_update()
        ).one()
        job = session.scalar(
            select(TrainingJob).where(TrainingJob.id == identifier).with_for_update()
        )
        if job.status != "running" or job.cancel_requested_at is not None:
            raise Cancelled()
        last = session.scalar(
            select(func.max(Model.version)).where(Model.project_id == project_id)
        )
        model = Model(
            project_id=project_id,
            job_id=identifier,
            version=(last or 0) + 1,
            metrics={**metrics, "artifacts": artifacts},
            class_map=class_map,
            weights_key=artifacts["run/best.pt"]["key"],
            onnx_key=artifacts["onnx/model.onnx"]["key"],
            input_size=input_size,
        )
        session.add(model)
        job.status, job.finished_at = (
            "succeeded",
            session.scalar(select(func.clock_timestamp())),
        )
        session.flush()
        model_id = model.id
    return str(model_id)


def finish_unsuccessful(sessions, identifier, error):
    with sessions.begin() as session:
        job = session.scalar(
            select(TrainingJob).where(TrainingJob.id == identifier).with_for_update()
        )
        if job.status == "running":
            job.status = (
                "cancelled" if job.cancel_requested_at is not None else "failed"
            )
            job.error = None if job.status == "cancelled" else error
            job.finished_at = session.scalar(select(func.clock_timestamp()))


def run_job(
    identifier,
    task_id,
    sessions,
    settings,
    datasets,
    images,
    artifacts,
    *,
    executor=execute,
):
    identifier = UUID(str(identifier))
    # Avoid blocking a duplicate delivery behind a busy GPU.
    with sessions() as session:
        row = session.get(TrainingJob, identifier)
        if row is None or row.status != "queued" or row.celery_task_id != task_id:
            return None
    with gpu_slot(settings) as root:
        claimed = claim(sessions, identifier, task_id, settings.worker_gpu_id)
        if claimed is None:
            return None
        job, version, task = claimed
        stage = "Dataset verification failed"
        try:
            request = TypeAdapter(JobCreate).validate_python(
                dict(
                    dataset_version_id=version.id,
                    model_arch=job.model_arch,
                    hyperparams=job.hyperparams,
                )
            )
            with TemporaryDirectory(prefix=f"{identifier}-", dir=root) as directory:
                work = Path(directory)

                def check():
                    check_active(sessions, identifier)

                manifest = materialize(
                    version, job.project_id, task, work, datasets, images, check
                )
                stage = "Training or export verification failed; inspect the private worker log"
                epoch = 0

                def progress():
                    nonlocal epoch
                    epoch = record_history(
                        sessions, identifier, work / "run/history.json", epoch
                    )

                executor(
                    dict(
                        work=str(work),
                        model_arch=job.model_arch,
                        hyperparams=request.hyperparams.model_dump(),
                        device=settings.worker_device,
                        cpu_threads=settings.worker_cpu_threads,
                        cache=str(settings.worker_cache_dir.resolve()),
                    ),
                    settings,
                    root / f"{identifier}.log",
                    check,
                    progress,
                )
                check()
                metrics, verification, input_size = verify_bundle(
                    work, manifest, version.counts["manifest_sha256"]
                )
                stage = "Artifact publication failed"
                prefix = f"{job.project_id}/jobs/{identifier}/{uuid4()}"
                index = {}
                for name in (
                    "manifest.json",
                    "audit.json",
                    "run/best.pt",
                    "run/run.json",
                    "run/history.json",
                    "evaluation.json",
                    "onnx/model.onnx",
                    "onnx/metadata.json",
                    "onnx/verification.json",
                ):
                    check()
                    index[name] = artifacts.put_file_once(
                        f"{prefix}/{name}", work / name
                    )
                return register(
                    sessions,
                    identifier,
                    job.project_id,
                    dict(manifest.class_map),
                    {
                        "test": metrics,
                        "parity": verification["parity"],
                        "latency": verification["latency"],
                    },
                    index,
                    input_size,
                )
        except Cancelled:
            finish_unsuccessful(sessions, identifier, "Training stopped")
        except Exception:
            # Never expose exception text containing credentials, hostnames or local paths.
            try:
                descriptor = os.open(
                    root / f"{identifier}.log",
                    os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                    0o600,
                )
                with os.fdopen(descriptor, "w") as log:
                    traceback.print_exc(file=log)
            except OSError:
                pass  # Logging failure must not prevent recording a terminal job state.
            finish_unsuccessful(sessions, identifier, stage)
        return None

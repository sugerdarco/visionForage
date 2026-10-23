"""Typed public evaluation views of checksummed worker reports."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, ValidationError

from vf_api.model_artifacts import artifact_record, read_report, unavailable

Probability = Annotated[FiniteFloat, Field(ge=0, le=1)]
Nonnegative = Annotated[FiniteFloat, Field(ge=0)]
Count = Annotated[int, Field(ge=0)]


class ReportFields(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")


class ClassScores(ReportFields):
    support: Count
    precision: Probability
    recall: Probability
    f1: Probability
    roc_auc: Probability | None


class ConfusionMatrix(ReportFields):
    class_names: list[str]
    rows: Literal["true"]
    columns: Literal["predicted"]
    counts: list[list[Count]]


class ClassificationMetrics(ReportFields):
    samples: Annotated[int, Field(gt=0)]
    accuracy: Probability
    loss: Nonnegative
    macro_f1: Probability
    roc_auc: Probability | None
    roc_auc_method: Literal["binary", "macro_one_vs_rest"]
    positive_class: str | None
    zero_division: Literal[0]
    per_class: dict[str, ClassScores]
    confusion_matrix: ConfusionMatrix


class DetectionCounts(ReportFields):
    true_positives: Count
    false_positives: Count
    false_negatives: Count
    support: Count
    predictions: Count
    precision: Probability
    recall: Probability


class DetectionClass(DetectionCounts):
    class_index: Count
    class_name: str
    ap50: Probability | None
    ap50_95: Probability | None


class DetectionMetrics(ReportFields):
    map50: Probability | None
    map50_95: Probability | None
    fixed_threshold_micro: DetectionCounts
    per_class: list[DetectionClass]


class DetectionSettings(ReportFields):
    ap_confidence_floor: Probability
    ap_iou_thresholds: list[Probability]
    precision_recall_confidence: Probability
    precision_recall_iou: Probability
    nms_iou: Probability
    max_detections_per_image: Annotated[int, Field(gt=0)]


class ParityReport(ReportFields):
    passed: Literal[True]
    split: Literal["test"]
    samples: Annotated[int, Field(ge=32)]
    batch_sizes: list[int]
    absolute_tolerance: Nonnegative
    relative_tolerance: Literal[0.0]
    max_absolute_error: Nonnegative
    box_absolute_tolerance: Nonnegative | None = None
    score_absolute_tolerance: Nonnegative | None = None
    max_box_absolute_error: Nonnegative | None = None
    max_score_absolute_error: Nonnegative | None = None


class LatencyReport(ReportFields):
    batch_size: Annotated[int, Field(gt=0)]
    warmup_runs: Count
    measured_runs: Annotated[int, Field(gt=0)]
    mean_ms: Nonnegative
    p95_ms: Nonnegative


class EvaluationBase(ReportFields):
    model_id: UUID
    split: Literal["test"]
    checkpoint_epoch: Annotated[int, Field(ge=0)]
    class_map: dict[str, int]
    parity: ParityReport
    latency: LatencyReport
    latency_scope: Literal["worker_cpu_model_only"] = "worker_cpu_model_only"


class ClassificationEvaluation(EvaluationBase):
    task_type: Literal["classification"]
    selection_val_loss: Nonnegative
    metrics: ClassificationMetrics


class DetectionEvaluation(EvaluationBase):
    task_type: Literal["detection"]
    selection_split: Literal["val"]
    metric_settings: DetectionSettings
    metrics: DetectionMetrics


EvaluationResponse = Annotated[
    ClassificationEvaluation | DetectionEvaluation, Field(discriminator="task_type")
]


def evaluation_response(row, task_type, store):
    evaluation = read_report(store, row, "evaluation.json")
    verification = read_report(store, row, "onnx/verification.json")
    if (
        evaluation.get("task_type") != task_type
        or evaluation.get("split") != "test"
        or evaluation.get("smoke_run", False) is not False
        or verification.get("smoke_run", False) is not False
        or evaluation.get("manifest_sha256")
        != artifact_record(row, "manifest.json")["sha256"]
        or evaluation.get("checkpoint_sha256")
        != artifact_record(row, "run/best.pt")["sha256"]
        or verification.get("model_sha256")
        != artifact_record(row, "onnx/model.onnx")["sha256"]
    ):
        raise unavailable()
    schema = {
        "classification": ClassificationEvaluation,
        "detection": DetectionEvaluation,
    }.get(task_type)
    if schema is None:
        raise unavailable()
    try:
        result = schema.model_validate(
            {
                **evaluation,
                "model_id": row.id,
                "class_map": row.class_map,
                "parity": verification.get("parity"),
                "latency": verification.get("latency"),
                "latency_scope": "worker_cpu_model_only",
            }
        )
    except ValidationError:
        raise unavailable() from None
    names = [
        name for name, _ in sorted(result.class_map.items(), key=lambda item: item[1])
    ]
    if sorted(result.class_map.values()) != list(range(len(names))):
        raise unavailable()
    parity = result.parity
    if parity.batch_sizes != [1, 8]:
        raise unavailable()
    if task_type == "classification":
        matrix = result.metrics.confusion_matrix
        if (
            set(result.metrics.per_class) != set(names)
            or matrix.class_names != names
            or len(matrix.counts) != len(names)
            or any(len(r) != len(names) for r in matrix.counts)
            or sum(map(sum, matrix.counts)) != result.metrics.samples
            or parity.absolute_tolerance != 0.001
            or parity.max_absolute_error > 0.001
        ):
            raise unavailable()
    else:
        if (
            evaluation.get("class_map") != row.class_map
            or [(c.class_name, c.class_index) for c in result.metrics.per_class]
            != list(zip(names, range(len(names))))
            or parity.absolute_tolerance != 0.002
            or parity.box_absolute_tolerance != 0.002
            or parity.score_absolute_tolerance != 0.001
            or parity.max_box_absolute_error is None
            or parity.max_box_absolute_error > 0.002
            or parity.max_score_absolute_error is None
            or parity.max_score_absolute_error > 0.001
        ):
            raise unavailable()
    return result

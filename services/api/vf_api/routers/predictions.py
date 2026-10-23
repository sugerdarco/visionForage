"""Owner-scoped history and SQL aggregates over recorded inference results."""

from datetime import timedelta
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Query
from sqlalchemy import func, select, text

from vf_api.dependencies import CurrentUser, Database
from vf_api.models import Model, Prediction, Project
from vf_api.prediction_schemas import DeploymentStats, PredictionPage, PredictionView
from vf_api.routers.deployments import owned_deployment
from vf_api.upload_dependencies import Predictions

router = APIRouter(tags=["predictions"])


@router.get("/deployments/{deployment_id}/predictions", response_model=PredictionPage)
def list_predictions(
    deployment_id: UUID,
    session: Database,
    user: CurrentUser,
    store: Predictions,
    page: Annotated[int, Query(ge=1, le=1_000_000)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
):
    deployment, _, version = owned_deployment(deployment_id, session, user)
    condition = Prediction.deployment_id == deployment_id
    total = session.scalar(
        select(func.count()).select_from(Prediction).where(condition)
    )
    rows = session.scalars(
        select(Prediction)
        .where(condition)
        .order_by(Prediction.created_at.desc(), Prediction.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return PredictionPage(
        items=[
            PredictionView(
                id=row.id,
                deployment_id=deployment_id,
                model_id=deployment.model_id,
                model_version=version,
                output=row.output,
                max_confidence=row.max_confidence,
                latency_ms=row.latency_ms,
                sent_to_review=row.sent_to_review,
                created_at=row.created_at,
                image_url=store.url(row.image_key),
            )
            for row in rows
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/deployments/{deployment_id}/stats", response_model=DeploymentStats)
def deployment_stats(
    deployment_id: UUID,
    session: Database,
    user: CurrentUser,
    window: Literal["1h", "24h", "7d", "30d"] = "24h",
):
    deployment, project_id, _ = owned_deployment(deployment_id, session, user)
    end = session.scalar(select(func.clock_timestamp()))
    start = end - timedelta(hours={"1h": 1, "24h": 24, "7d": 168, "30d": 720}[window])
    params = {"deployment": deployment_id, "start": start, "end": end}
    # One statement/snapshot keeps all counts consistent while the collector writes.
    result = (
        session.execute(
            text("""
        WITH selected AS MATERIALIZED (
            SELECT output, max_confidence, latency_ms, sent_to_review
            FROM predictions WHERE deployment_id = :deployment
            AND created_at >= :start AND created_at < :end
        ), classes AS (
            SELECT output->>'label' AS label FROM selected WHERE output ? 'label'
            UNION ALL
            SELECT box->>'label' FROM selected
            CROSS JOIN LATERAL jsonb_array_elements(COALESCE(output->'boxes', '[]'::jsonb)) AS box
        ), class_counts AS (
            SELECT label, count(*) AS n FROM classes GROUP BY label
        ), bins AS (
            SELECT LEAST(9, floor(max_confidence * 10)::int) AS bin, count(*) AS n
            FROM selected GROUP BY bin
        )
        SELECT count(*) AS requests,
            count(*) FILTER (WHERE sent_to_review) AS low_confidence_count,
            avg(latency_ms) AS mean,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms) AS p50,
            percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95,
            (SELECT jsonb_object_agg(label, n) FROM class_counts) AS classes,
            (SELECT jsonb_object_agg(bin, n) FROM bins) AS bins
        FROM selected
    """),
            params,
        )
        .mappings()
        .one()
    )
    class_map = session.scalar(
        select(Model.class_map).where(Model.id == deployment.model_id)
    )
    task = session.scalar(select(Project.task_type).where(Project.id == project_id))
    classes, bins = result["classes"] or {}, result["bins"] or {}
    return DeploymentStats(
        deployment_id=deployment_id,
        window=window,
        start=start,
        end=end,
        requests=result["requests"],
        low_confidence_count=result["low_confidence_count"],
        latency_mean_ms=result["mean"],
        latency_p50_ms=result["p50"],
        latency_p95_ms=result["p95"],
        class_distribution={label: classes.get(label, 0) for label in class_map},
        class_count_unit="boxes" if task == "detection" else "predictions",
        confidence_histogram=[
            {
                "lower": i / 10,
                "upper": (i + 1) / 10,
                "upper_inclusive": i == 9,
                "count": bins.get(str(i), 0),
            }
            for i in range(10)
        ],
    )

import type { components } from "./schema";

type Schemas = components["schemas"];

export type User = Schemas["UserResponse"];
export type LoginResponse = Schemas["LoginResponse"];
export type AccessTokenResponse = Schemas["AccessTokenResponse"];
export type Project = Schemas["ProjectResponse"];
export type ProjectDetail = Schemas["ProjectDetail"];
export type ProjectPage = Schemas["ProjectPage"];
export type ProjectCreate = Schemas["ProjectCreate"];
export type ClassCreate = Schemas["ClassCreate"];
export type ProjectClass = Schemas["ClassResponse"];
export type FileClaim = Schemas["FileClaim"];
export type UploadPermission = Schemas["UploadPermission"];
export type Image = Schemas["ImageResponse"];
export type ImagePage = Schemas["ImagePage"];
export type ImageStatus = Image["status"];
export type ImageAnnotations = Schemas["ImageAnnotations"];
export type Annotation = Schemas["AnnotationResponse"];
export type AnnotationInput = Schemas["AnnotationInput"];
export type AnnotationSave = Schemas["AnnotationSave"];
export type ImportClaim = Schemas["ImportClaim"];
export type ImportResponse = Schemas["ImportResponse"];
export type ImportFormat = ImportClaim["format"];
export type DatasetVersion = Schemas["VersionResponse"];
export type VersionPage = Schemas["VersionPage"];
export type VersionCreate = Schemas["VersionCreate"];
export type Job = Schemas["JobResponse"];
export type JobPage = Schemas["JobPage"];
export type JobStatus = Job["status"];
export type ClassificationJob = Schemas["ClassificationJob"];
export type DetectionJob = Schemas["DetectionJob"];
export type ClassificationParameters = Schemas["ClassificationParameters"];
export type DetectionParameters = Schemas["DetectionParameters"];
export type Metric = Schemas["MetricResponse"];
export type MetricPage = Schemas["MetricPage"];
export type Model = Schemas["ModelResponse"];
export type ModelDetail = Schemas["ModelDetail"];
export type ModelPage = Schemas["ModelPage"];
export type Evaluation = Schemas["ClassificationEvaluation"] | Schemas["DetectionEvaluation"];
export type Artifact = Schemas["ArtifactResponse"];
export type ArtifactDownload = Schemas["ArtifactDownload"];
export type ValidationIssue = Schemas["ValidationError"];

export type TaskType = "classification" | "detection";
export type Split = "train" | "val" | "test";

/**
 * The version `counts` column is untyped JSON in OpenAPI. This is the shape the
 * builder writes (services/api/vf_api/dataset_snapshots.py); read it defensively.
 */
export interface VersionCounts {
  images?: Partial<Record<Split, number>>;
  labels?: Partial<Record<Split, number>>;
  pixel_groups?: Partial<Record<Split, number>>;
  classes?: Partial<Record<Split, Record<string, number>>>;
  excluded?: Record<string, number>;
  seed?: number;
  split_method?: string;
  duplicate_groups?: number;
  class_map?: Record<string, number>;
  warnings?: { split: string; missing_classes: string[] }[];
  manifest_sha256?: string;
  audit_sha256?: string;
}

export function versionCounts(version: DatasetVersion): VersionCounts {
  return version.counts as VersionCounts;
}

export const TERMINAL_JOB_STATUSES: readonly JobStatus[] = ["succeeded", "failed", "cancelled"];

export function isTerminal(status: JobStatus): boolean {
  return TERMINAL_JOB_STATUSES.includes(status);
}

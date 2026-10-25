import { apiClient, type ApiClient } from "./client";
import type {
  AnnotationSave,
  Deployment,
  DeploymentCreated,
  DeploymentCreate,
  DeploymentPage,
  PredictionPage,
  DeploymentStats,
  ReviewAccept,
  ReviewReceipt,
  Evaluation,
  Artifact,
  ArtifactDownload,
  ClassCreate,
  ClassificationJob,
  DatasetVersion,
  DetectionJob,
  FileClaim,
  Image,
  ImageAnnotations,
  ImagePage,
  ImageStatus,
  ImportClaim,
  ImportResponse,
  Job,
  JobPage,
  LoginResponse,
  Metric,
  MetricPage,
  ModelDetail,
  ModelPage,
  ProjectClass,
  ProjectCreate,
  ProjectDetail,
  ProjectPage,
  UploadPermission,
  User,
  VersionCreate,
  VersionPage,
} from "./types";

const MAX_PAGE_SIZE = 100;

/** Typed calls for every implemented route. Paths mirror services/api/vf_api/routers. */
export function createApi(client: ApiClient) {
  return {
    auth: {
      register: (email: string, password: string) =>
        client.post<User>("/auth/register", { email, password }, { auth: false }),
      login: (email: string, password: string) =>
        client
          .post<LoginResponse>("/auth/login", { email, password }, { auth: false })
          .then((r) => r.data),
      me: (signal?: AbortSignal) => client.get<User>("/auth/me", { signal }),
    },
    projects: {
      list: (page: number, pageSize = 20, signal?: AbortSignal) =>
        client.get<ProjectPage>("/projects", { query: { page, page_size: pageSize }, signal }),
      create: (body: ProjectCreate) =>
        client.post<ProjectDetail>("/projects", body).then((r) => r.data),
      get: (projectId: string, signal?: AbortSignal) =>
        client.get<ProjectDetail>(`/projects/${projectId}`, { signal }),
      rename: (projectId: string, name: string) =>
        client
          .request<ProjectDetail>("PATCH", `/projects/${projectId}`, { body: { name } })
          .then((r) => r.data),
      remove: (projectId: string) =>
        client.request<void>("DELETE", `/projects/${projectId}`).then(() => undefined),
      classes: (projectId: string, signal?: AbortSignal) =>
        client.get<ProjectClass[]>(`/projects/${projectId}/classes`, { signal }),
      addClasses: (projectId: string, classes: ClassCreate[]) =>
        client
          .post<ProjectClass[]>(`/projects/${projectId}/classes`, { classes })
          .then((r) => r.data),
    },
    uploads: {
      presign: (projectId: string, files: FileClaim[]) =>
        client
          .post<UploadPermission[]>(`/projects/${projectId}/uploads/presign`, { files })
          .then((r) => r.data),
      /** 201 registers a new image; 200 means identical bytes already exist in the project. */
      complete: (projectId: string, uploadToken: string) =>
        client.post<Image>(`/projects/${projectId}/uploads/complete`, {
          upload_token: uploadToken,
        }),
    },
    images: {
      list: (
        projectId: string,
        params: {
          page: number;
          pageSize?: number;
          status?: ImageStatus | null;
          classId?: string | null;
        },
        signal?: AbortSignal,
      ) =>
        client.get<ImagePage>(`/projects/${projectId}/images`, {
          query: {
            page: params.page,
            page_size: params.pageSize ?? 20,
            status: params.status,
            class: params.classId,
          },
          signal,
        }),
      get: (imageId: string, signal?: AbortSignal) =>
        client.get<Image>(`/images/${imageId}`, { signal }),
      annotations: (imageId: string, signal?: AbortSignal) =>
        client.get<ImageAnnotations>(`/images/${imageId}/annotations`, { signal }),
      saveAnnotations: (imageId: string, body: AnnotationSave, signal?: AbortSignal) =>
        client
          .request<ImageAnnotations>("PUT", `/images/${imageId}/annotations`, { body, signal })
          .then((r) => r.data),
    },
    imports: {
      presign: (projectId: string, claim: ImportClaim) =>
        client
          .post<UploadPermission>(`/projects/${projectId}/imports/presign`, claim)
          .then((r) => r.data),
      complete: (projectId: string, uploadToken: string) =>
        client
          .post<ImportResponse>(`/projects/${projectId}/imports`, { upload_token: uploadToken })
          .then((r) => r.data),
    },
    versions: {
      list: (projectId: string, page: number, pageSize = 20, signal?: AbortSignal) =>
        client.get<VersionPage>(`/projects/${projectId}/dataset-versions`, {
          query: { page, page_size: pageSize },
          signal,
        }),
      get: (projectId: string, versionId: string, signal?: AbortSignal) =>
        client.get<DatasetVersion>(`/projects/${projectId}/dataset-versions/${versionId}`, {
          signal,
        }),
      /** Not idempotent: every successful call creates another version. */
      create: (projectId: string, body: VersionCreate) =>
        client
          .post<DatasetVersion>(`/projects/${projectId}/dataset-versions`, body)
          .then((r) => r.data),
    },
    jobs: {
      list: (projectId: string, page: number, pageSize = 20, signal?: AbortSignal) =>
        client.get<JobPage>(`/projects/${projectId}/jobs`, {
          query: { page, page_size: pageSize },
          signal,
        }),
      get: (jobId: string, signal?: AbortSignal) => client.get<Job>(`/jobs/${jobId}`, { signal }),
      /** Not idempotent: every successful call queues another training run. */
      create: (projectId: string, body: ClassificationJob | DetectionJob) =>
        client.post<Job>(`/projects/${projectId}/jobs`, body).then((r) => r.data),
      cancel: (jobId: string) => client.post<Job>(`/jobs/${jobId}/cancel`).then((r) => r.data),
      metricsPage: (jobId: string, page: number, signal?: AbortSignal) =>
        client.get<MetricPage>(`/jobs/${jobId}/metrics`, {
          query: { page, page_size: MAX_PAGE_SIZE },
          signal,
        }),
      /** Read every metrics page; history is incomplete until the last page arrives. */
      allMetrics: async (jobId: string, signal?: AbortSignal): Promise<Metric[]> => {
        const rows: Metric[] = [];
        for (let page = 1; ; page++) {
          const result = await client.get<MetricPage>(`/jobs/${jobId}/metrics`, {
            query: { page, page_size: MAX_PAGE_SIZE },
            signal,
          });
          rows.push(...result.items);
          if (result.items.length === 0 || page * result.page_size >= result.total) return rows;
        }
      },
    },
    deployments: {
      list: (projectId: string, page = 1, signal?: AbortSignal) =>
        client.get<DeploymentPage>(`/projects/${projectId}/deployments`, {
          query: { page, page_size: 100 },
          signal,
        }),
      get: (id: string, signal?: AbortSignal) =>
        client.get<Deployment>(`/deployments/${id}`, { signal }),
      create: (modelId: string, body: DeploymentCreate) =>
        client.post<DeploymentCreated>(`/models/${modelId}/deployments`, body).then((r) => r.data),
      stop: (id: string) =>
        client.request<Deployment>("DELETE", `/deployments/${id}`).then((r) => r.data),
      stats: (id: string, window: DeploymentStats["window"], signal?: AbortSignal) =>
        client.get<DeploymentStats>(`/deployments/${id}/stats`, { query: { window }, signal }),
      predictions: (id: string, page: number, signal?: AbortSignal) =>
        client.get<PredictionPage>(`/deployments/${id}/predictions`, {
          query: { page, page_size: 20 },
          signal,
        }),
    },
    reviews: {
      list: (projectId: string, page: number, signal?: AbortSignal) =>
        client.get<PredictionPage>(`/projects/${projectId}/review-queue`, {
          query: { page, page_size: 20 },
          signal,
        }),
      accept: (id: string, body: ReviewAccept) =>
        client.post<ReviewReceipt>(`/review/${id}/accept`, body).then((r) => r.data),
    },
    models: {
      evaluation: (modelId: string, signal?: AbortSignal) =>
        client.get<Evaluation>(`/models/${modelId}/evaluation`, { signal }),
      artifacts: (modelId: string, signal?: AbortSignal) =>
        client.get<Artifact[]>(`/models/${modelId}/artifacts`, { signal }),
      download: (modelId: string, artifact: Artifact["name"]) =>
        client.get<ArtifactDownload>(`/models/${modelId}/artifacts/${artifact}`),
      promote: (modelId: string) =>
        client.post<ModelDetail>(`/models/${modelId}/promote`).then((r) => r.data),
      list: (projectId: string, page: number, pageSize = 20, signal?: AbortSignal) =>
        client.get<ModelPage>(`/projects/${projectId}/models`, {
          query: { page, page_size: pageSize },
          signal,
        }),
      get: (modelId: string, signal?: AbortSignal) =>
        client.get<ModelDetail>(`/models/${modelId}`, { signal }),
    },
  };
}

export const api = createApi(apiClient);
export type Api = ReturnType<typeof createApi>;

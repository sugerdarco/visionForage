import { API_BASE } from "@/lib/config";
import { ApiError, errorFromResponse, fallbackMessage, toApiError } from "./errors";
import type { AccessTokenResponse } from "./types";

export interface SessionTokens {
  accessToken: string;
  refreshToken: string;
}

type QueryValue = string | number | boolean | null | undefined;

export interface RequestOptions {
  body?: unknown;
  query?: Record<string, QueryValue>;
  signal?: AbortSignal;
  /** Send the bearer token and refresh once on 401. Defaults to true. */
  auth?: boolean;
}

export interface ApiResponse<T> {
  data: T;
  status: number;
}

interface ClientOptions {
  baseUrl?: string;
  fetchImpl?: typeof fetch;
  /** Called after a refresh attempt proves the session is no longer valid. */
  onSessionExpired?: () => void;
}

/**
 * The single REST wrapper. Tokens live only in memory. The bearer header is
 * attached only to requests built from `baseUrl`; signed storage requests use
 * `lib/api/storage.ts` instead so authorization cannot leak to MinIO.
 */
export class ApiClient {
  private tokens: SessionTokens | null = null;
  private sessionVersion = 0;
  private refreshing: Promise<string> | null = null;
  private readonly baseUrl: string;
  private readonly fetchImpl: typeof fetch;
  private onSessionExpired: (() => void) | undefined;

  constructor(options: ClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? API_BASE).replace(/\/+$/, "");
    this.fetchImpl = options.fetchImpl ?? ((...args) => globalThis.fetch(...args));
    this.onSessionExpired = options.onSessionExpired;
  }

  setSessionExpiredHandler(handler: (() => void) | undefined) {
    this.onSessionExpired = handler;
  }

  setTokens(tokens: SessionTokens | null) {
    this.sessionVersion++;
    this.tokens = tokens;
    this.refreshing = null;
  }

  hasSession(): boolean {
    return this.tokens !== null;
  }

  get accessToken(): string | null {
    return this.tokens?.accessToken ?? null;
  }

  url(path: string, query?: Record<string, QueryValue>): string {
    const url = new URL(`${this.baseUrl}${path}`);
    for (const [key, value] of Object.entries(query ?? {})) {
      if (value !== undefined && value !== null && value !== "") {
        url.searchParams.set(key, String(value));
      }
    }
    return url.toString();
  }

  async request<T>(
    method: string,
    path: string,
    options: RequestOptions = {},
  ): Promise<ApiResponse<T>> {
    const auth = options.auth ?? true;
    const sessionVersion = this.sessionVersion;
    const checkSession = () => {
      if (auth && sessionVersion !== this.sessionVersion) throw this.sessionError();
    };
    const token = auth ? this.requireToken() : null;
    let response = await this.send(method, path, options, token);
    checkSession();
    if (auth && response.status === 401) {
      // A 401 means the API rejected the credential before running the handler,
      // so repeating the same request once with a fresh token is safe.
      const fresh = await this.refreshAccessToken(token);
      checkSession();
      response = await this.send(method, path, options, fresh);
      checkSession();
      if (response.status === 401) {
        this.expire();
        throw await errorFromResponse(response);
      }
    }
    if (!response.ok) throw await errorFromResponse(response);
    const data = (response.status === 204 ? undefined : await response.json()) as T;
    checkSession();
    return { data, status: response.status };
  }

  get<T>(path: string, options?: Omit<RequestOptions, "body">) {
    return this.request<T>("GET", path, options).then((r) => r.data);
  }

  post<T>(path: string, body?: unknown, options?: Omit<RequestOptions, "body">) {
    return this.request<T>("POST", path, { ...options, body });
  }

  /**
   * Exchange the refresh token once, even if many requests notice expiry together.
   * `staleToken` is the access token that failed; if another caller already replaced
   * it, the newer token is returned without another refresh request.
   */
  async refreshAccessToken(staleToken: string | null): Promise<string> {
    const current = this.tokens;
    if (!current) throw this.sessionError();
    if (staleToken !== null && current.accessToken !== staleToken) return current.accessToken;
    if (!this.refreshing) {
      const refreshToken = current.refreshToken;
      const pending = this.exchange(refreshToken, this.sessionVersion).finally(() => {
        if (this.refreshing === pending) this.refreshing = null;
      });
      this.refreshing = pending;
    }
    return this.refreshing;
  }

  private async exchange(refreshToken: string, sessionVersion: number): Promise<string> {
    let response: Response;
    try {
      response = await this.fetchImpl(this.url("/auth/refresh"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: refreshToken }),
        cache: "no-store",
      });
    } catch (error) {
      throw toApiError(error);
    }
    if (sessionVersion !== this.sessionVersion) throw this.sessionError();
    if (response.status === 401 || response.status === 422) {
      this.expire();
      throw this.sessionError();
    }
    if (!response.ok) throw await errorFromResponse(response);
    const body = (await response.json()) as AccessTokenResponse;
    // A sign-out or another login may have happened while this request was in flight.
    if (sessionVersion !== this.sessionVersion || this.tokens?.refreshToken !== refreshToken)
      throw this.sessionError();
    this.tokens = { refreshToken, accessToken: body.access_token };
    return body.access_token;
  }

  private expire() {
    if (!this.tokens) return;
    this.tokens = null;
    this.sessionVersion++;
    this.refreshing = null;
    this.onSessionExpired?.();
  }

  private requireToken(): string {
    if (!this.tokens) throw this.sessionError();
    return this.tokens.accessToken;
  }

  private sessionError() {
    return new ApiError({
      kind: "unauthorized",
      status: 401,
      message: fallbackMessage("unauthorized"),
    });
  }

  private async send(
    method: string,
    path: string,
    options: RequestOptions,
    token: string | null,
  ): Promise<Response> {
    const headers: Record<string, string> = { Accept: "application/json" };
    if (options.body !== undefined) headers["Content-Type"] = "application/json";
    if (token) headers.Authorization = `Bearer ${token}`;
    try {
      return await this.fetchImpl(this.url(path, options.query), {
        method,
        headers,
        body: options.body === undefined ? undefined : JSON.stringify(options.body),
        signal: options.signal,
        cache: "no-store",
      });
    } catch (error) {
      throw toApiError(error);
    }
  }
}

export const apiClient = new ApiClient();

import { describe, expect, it, vi } from "vitest";
import { ApiClient } from "@/lib/api/client";
import { errorFromResponse } from "@/lib/api/errors";
import { submitOnce } from "@/lib/api/outcome";

function json(status: number, body: unknown, headers: Record<string, string> = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  });
}

describe("ApiClient authentication", () => {
  it("refreshes once for concurrent 401s and retries each request with the new token", async () => {
    let refreshes = 0;
    const fetchImpl = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/auth/refresh")) {
        refreshes++;
        await new Promise((r) => setTimeout(r, 5));
        return json(200, { access_token: "new", token_type: "bearer", expires_in: 900 });
      }
      const auth = (init?.headers as Record<string, string>).Authorization;
      return auth === "Bearer new" ? json(200, { ok: url }) : json(401, { detail: "expired" });
    });
    const client = new ApiClient({ baseUrl: "http://api.test/api/v1", fetchImpl });
    client.setTokens({ accessToken: "old", refreshToken: "r" });
    const results = await Promise.all([client.get("/a"), client.get("/b"), client.get("/c")]);
    expect(refreshes).toBe(1);
    expect(results).toHaveLength(3);
    expect(client.accessToken).toBe("new");
  });

  it("signs out when refresh is rejected", async () => {
    const expired = vi.fn();
    const fetchImpl = vi.fn(async (input: RequestInfo | URL) =>
      String(input).endsWith("/auth/refresh")
        ? json(401, { detail: "Invalid token" })
        : json(401, { detail: "expired" }),
    );
    const client = new ApiClient({
      baseUrl: "http://api.test/api/v1",
      fetchImpl,
      onSessionExpired: expired,
    });
    client.setTokens({ accessToken: "old", refreshToken: "r" });
    await expect(client.get("/projects")).rejects.toMatchObject({ kind: "unauthorized" });
    expect(expired).toHaveBeenCalledTimes(1);
    expect(client.hasSession()).toBe(false);
  });

  it("does not resurrect a session that signed out during refresh", async () => {
    let release!: () => void;
    const fetchImpl = vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).endsWith("/auth/refresh")) {
        await new Promise<void>((r) => (release = r));
        return json(200, { access_token: "new", token_type: "bearer", expires_in: 900 });
      }
      return json(401, {});
    });
    const client = new ApiClient({ baseUrl: "http://api.test/api/v1", fetchImpl });
    client.setTokens({ accessToken: "old", refreshToken: "r" });
    const pending = client.get("/projects");
    await new Promise((r) => setTimeout(r, 0));
    client.setTokens(null);
    release();
    await expect(pending).rejects.toMatchObject({ kind: "unauthorized" });
    expect(client.hasSession()).toBe(false);
  });
});

describe("error parsing", () => {
  it("reads validation arrays and safe string details", async () => {
    const validation = await errorFromResponse(
      json(422, {
        detail: [
          {
            loc: ["body", "password"],
            msg: "String should have at least 12 characters",
            type: "x",
          },
        ],
      }),
    );
    expect(validation.kind).toBe("validation");
    expect(validation.fieldErrors()).toEqual({
      password: "String should have at least 12 characters",
    });
    const conflict = await errorFromResponse(json(409, { detail: "Class names already exist" }));
    expect(conflict).toMatchObject({ kind: "conflict", message: "Class names already exist" });
  });

  it("does not reveal whether a 404 resource exists", async () => {
    const missing = await errorFromResponse(json(404, { detail: "Project not found" }));
    expect(missing.message).toMatch(/not found or is not available/);
  });
});

describe("non-idempotent submissions", () => {
  it("treats a lost response as uncertain rather than failed", async () => {
    const outcome = await submitOnce(() => Promise.reject(new TypeError("Failed to fetch")));
    expect(outcome.kind).toBe("uncertain");
  });

  it("treats validation and conflicts as definite rejections", async () => {
    const error = await errorFromResponse(json(422, { detail: "Dataset needs 32 test images" }));
    expect((await submitOnce(() => Promise.reject(error))).kind).toBe("rejected");
  });

  it("treats gateway failures as uncertain", async () => {
    const error = await errorFromResponse(json(502, {}));
    expect((await submitOnce(() => Promise.reject(error))).kind).toBe("uncertain");
  });
});

describe("account changes during requests", () => {
  it("does not expire a new account when an old refresh is rejected", async () => {
    let resolve!: (response: Response) => void;
    const expired = vi.fn();
    const client = new ApiClient({
      baseUrl: "http://api.test/api/v1",
      onSessionExpired: expired,
      fetchImpl: vi.fn(
        () =>
          new Promise<Response>((r) => {
            resolve = r;
          }),
      ),
    });
    client.setTokens({ accessToken: "old", refreshToken: "old-refresh" });
    const pending = client.refreshAccessToken("old");
    client.setTokens({ accessToken: "other", refreshToken: "other-refresh" });
    resolve(json(401, {}));
    await expect(pending).rejects.toMatchObject({ kind: "unauthorized" });
    expect(client.accessToken).toBe("other");
    expect(expired).not.toHaveBeenCalled();
  });

  it("never retries an old mutation under a newly signed-in account", async () => {
    let resolve!: (response: Response) => void;
    const fetchImpl = vi.fn(
      () =>
        new Promise<Response>((r) => {
          resolve = r;
        }),
    );
    const client = new ApiClient({ baseUrl: "http://api.test/api/v1", fetchImpl });
    client.setTokens({ accessToken: "old", refreshToken: "old-refresh" });
    const pending = client.post("/projects", { name: "Old account project" });
    client.setTokens({ accessToken: "other", refreshToken: "other-refresh" });
    resolve(json(401, {}));
    await expect(pending).rejects.toMatchObject({ kind: "unauthorized" });
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(client.accessToken).toBe("other");
  });
});

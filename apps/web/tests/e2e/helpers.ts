import { expect, type Page } from "@playwright/test";
import { join } from "node:path";
import { generate } from "../fixtures/generate";

export const API = process.env.E2E_API_URL ?? "http://127.0.0.1:8000/api/v1";
export const PASSWORD = "correct horse battery staple";

/** Unique per run so repeated runs never collide with earlier accounts or image hashes. */
export const RUN = Date.now() % 1_000_000;
export const FIXTURES: string = generate(RUN);
export const fixture = (name: string) => join(FIXTURES, name);

export function uniqueEmail(tag: string) {
  return `e2e-${tag}-${RUN}-${Math.floor(Math.random() * 1e6)}@example.com`;
}

export async function register(page: Page, email: string) {
  const response = await page.request.post(`${API}/auth/register`, {
    data: { email, password: PASSWORD },
  });
  expect(response.status()).toBe(201);
}

export async function registerAndLogin(page: Page, email: string) {
  await page.goto("/register");
  await page.getByLabel("Email").fill(email);
  await page.getByLabel("Password", { exact: true }).fill(PASSWORD);
  await page.getByLabel("Confirm password").fill(PASSWORD);
  await page.getByRole("button", { name: "Create account" }).click();
  await expect(page).toHaveURL(/\/login\?registered=1/);
  await signInHere(page, email);
  await expect(page.getByRole("heading", { name: "Projects", exact: true })).toBeVisible();
}

export async function signInHere(page: Page, email: string) {
  await page.getByLabel("Email").fill(email);
  await page.getByLabel("Password").fill(PASSWORD);
  await page.getByRole("button", { name: "Sign in" }).click();
}

/**
 * Sessions live in memory, so a full navigation signs out. Deep links therefore
 * pass through the login form, which returns to the requested path.
 */
export async function openAuthed(page: Page, path: string, email: string) {
  await page.goto(path);
  await expect(page).toHaveURL(/\/login\?next=/);
  await signInHere(page, email);
  await expect(page).toHaveURL((url) => url.pathname + url.search === path);
}

export async function createProject(
  page: Page,
  name: string,
  task: "Classification" | "Object detection",
  classes: string[],
) {
  await page.getByRole("button", { name: "New project" }).click();
  const dialog = page.getByRole("dialog", { name: "New project" });
  await dialog.getByLabel("Project name").fill(name);
  await dialog.getByLabel(task).check();
  for (let i = 0; i < classes.length; i++) {
    if (i > 0) await dialog.getByRole("button", { name: "Add class" }).click();
    await dialog.getByLabel(`Name of class ${i + 1}`).fill(classes[i]!);
  }
  await dialog.getByRole("button", { name: "Create project" }).click();
  await expect(page).toHaveURL(/\/projects\/[0-9a-f-]+\/images/);
  return page.url().match(/projects\/([0-9a-f-]+)/)![1]!;
}

/** Direct API access for verifying what the browser stored (test-only token). */
export async function apiAs(page: Page, email: string) {
  const login = await page.request.post(`${API}/auth/login`, {
    data: { email, password: PASSWORD },
  });
  expect(login.ok()).toBeTruthy();
  const token = (await login.json()).access_token as string;
  const headers = { Authorization: `Bearer ${token}` };
  return {
    get: async (path: string) => {
      const r = await page.request.get(`${API}${path}`, { headers });
      expect(r.ok(), `${path} → ${r.status()}`).toBeTruthy();
      return r.json();
    },
    remove: (path: string) => page.request.delete(`${API}${path}`, { headers }),
    put: async (path: string, data: unknown) => {
      const r = await page.request.put(`${API}${path}`, { headers, data });
      expect(r.ok(), `${path} → ${r.status()}`).toBeTruthy();
      return r.json();
    },
    post: async (path: string, data?: unknown) =>
      page.request.post(`${API}${path}`, { headers, data }),
  };
}

export async function nav(page: Page, label: string) {
  await page
    .getByRole("navigation", { name: "Project" })
    .getByRole("link", { name: new RegExp(`^${label}`) })
    .click();
}

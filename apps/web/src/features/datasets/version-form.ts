import type { VersionCreate } from "@/lib/api/types";

export const MAX_SEED = 2 ** 32 - 1;
export const MIN_TEST_IMAGES_FOR_TRAINING = 32;

export interface VersionFormValues {
  seed: string;
  train: string;
  val: string;
  test: string;
}

export const DEFAULT_VERSION_FORM: VersionFormValues = {
  seed: "42",
  train: "70",
  val: "15",
  test: "15",
};

export type VersionFormErrors = Partial<Record<keyof VersionFormValues | "ratios", string>>;

/** Validate percentages and seed like the API, returning the JSON body when valid. */
export function parseVersionForm(values: VersionFormValues): {
  body?: VersionCreate;
  errors: VersionFormErrors;
} {
  const errors: VersionFormErrors = {};
  const seed = values.seed.trim();
  if (!/^\d+$/.test(seed) || Number(seed) > MAX_SEED) {
    errors.seed = `Use a whole number from 0 to ${MAX_SEED.toLocaleString("en-US")}.`;
  }
  const ratios: Record<"train" | "val" | "test", number> = { train: 0, val: 0, test: 0 };
  for (const split of ["train", "val", "test"] as const) {
    const raw = values[split].trim();
    const value = raw === "" ? Number.NaN : Number(raw);
    if (!Number.isFinite(value) || value <= 0 || value >= 100) {
      errors[split] = "Enter a percentage above 0 and below 100.";
    } else ratios[split] = value / 100;
  }
  if (!errors.train && !errors.val && !errors.test) {
    const sum = ratios.train + ratios.val + ratios.test;
    if (Math.abs(sum - 1) > 1e-9)
      errors.ratios = `The three splits must add up to 100% (now ${(sum * 100).toFixed(4).replace(/\.?0+$/, "")}%).`;
  }
  if (Object.keys(errors).length) return { errors };
  return { errors, body: { seed: Number(seed), split_ratios: ratios } };
}

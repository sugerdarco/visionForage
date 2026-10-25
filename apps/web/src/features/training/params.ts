import type { ClassificationJob, DetectionJob, TaskType } from "@/lib/api/types";

export const MAX_SEED = 2 ** 32 - 1;

export type ParamKind = "int" | "float" | "bool";

export interface ParamSpec {
  key: string;
  label: string;
  kind: ParamKind;
  min?: number;
  max?: number;
  /** Exclusive upper bound. */
  below?: number;
  multipleOf?: number;
  help: string;
}

const COMMON: ParamSpec[] = [
  {
    key: "epochs",
    label: "Epochs",
    kind: "int",
    min: 1,
    max: 1000,
    help: "Maximum passes over the training split. Early stopping can end sooner.",
  },
  {
    key: "batch_size",
    label: "Batch size",
    kind: "int",
    min: 1,
    max: 256,
    help: "Images per training step. Lower it if training runs out of GPU memory.",
  },
  {
    key: "patience",
    label: "Early-stopping patience",
    kind: "int",
    min: 1,
    max: 1000,
    help: "Stop after this many epochs without validation improvement.",
  },
  {
    key: "seed",
    label: "Seed",
    kind: "int",
    min: 0,
    max: MAX_SEED,
    help: "Fixes randomness so a run can be repeated.",
  },
  {
    key: "horizontal_flip",
    label: "Random horizontal flips",
    kind: "bool",
    help: "Only if a mirrored part is still a valid example.",
  },
  {
    key: "vertical_flip",
    label: "Random vertical flips",
    kind: "bool",
    help: "Only if an upside-down part is still a valid example.",
  },
];

export const PARAM_SPECS: Record<TaskType, ParamSpec[]> = {
  classification: [
    ...COMMON.slice(0, 4),
    {
      key: "learning_rate",
      label: "Learning rate",
      kind: "float",
      min: 1e-8,
      max: 1,
      help: "Peak AdamW learning rate (cosine schedule).",
    },
    {
      key: "weight_decay",
      label: "Weight decay",
      kind: "float",
      min: 0,
      max: 1,
      help: "AdamW regularization strength.",
    },
    {
      key: "label_smoothing",
      label: "Label smoothing",
      kind: "float",
      min: 0,
      below: 1,
      help: "Softens targets; 0 disables it.",
    },
    ...COMMON.slice(4),
  ],
  detection: [
    ...COMMON.slice(0, 4),
    {
      key: "image_size",
      label: "Image size (px)",
      kind: "int",
      min: 64,
      max: 1280,
      multipleOf: 32,
      help: "Training resolution; a multiple of 32.",
    },
    ...COMMON.slice(4),
  ],
};

export const DEFAULTS: Record<TaskType, Record<string, number | boolean>> = {
  classification: {
    epochs: 20,
    batch_size: 64,
    patience: 5,
    seed: 42,
    learning_rate: 0.0003,
    weight_decay: 0.01,
    label_smoothing: 0.1,
    horizontal_flip: false,
    vertical_flip: false,
  },
  detection: {
    epochs: 100,
    batch_size: 16,
    patience: 20,
    seed: 42,
    image_size: 640,
    horizontal_flip: false,
    vertical_flip: false,
  },
};

export const ARCHITECTURE: Record<TaskType, { id: "resnet18" | "yolov8s"; label: string }> = {
  classification: { id: "resnet18", label: "ResNet-18 (ImageNet pretrained)" },
  detection: { id: "yolov8s", label: "YOLOv8s (pretrained)" },
};

export type ParamValues = Record<string, string | boolean>;

export function initialValues(task: TaskType): ParamValues {
  return Object.fromEntries(
    Object.entries(DEFAULTS[task]).map(([k, v]) => [k, typeof v === "boolean" ? v : String(v)]),
  );
}

function checkNumber(spec: ParamSpec, raw: string): { value?: number; error?: string } {
  const text = raw.trim();
  if (text === "") return { error: "Enter a value." };
  if (spec.kind === "int" && !/^\d+$/.test(text)) return { error: "Enter a whole number." };
  const value = Number(text);
  if (!Number.isFinite(value)) return { error: "Enter a finite number." };
  const range =
    spec.below !== undefined
      ? `from ${spec.min} up to (but not including) ${spec.below}`
      : `from ${spec.min} to ${spec.max}`;
  if (
    (spec.min !== undefined && value < spec.min) ||
    (spec.max !== undefined && value > spec.max) ||
    (spec.below !== undefined && value >= spec.below)
  ) {
    return { error: `Use a value ${range}.` };
  }
  if (spec.multipleOf && value % spec.multipleOf !== 0)
    return { error: `Use a multiple of ${spec.multipleOf}.` };
  return { value };
}

/** Validate like the API and build the JSON body with numbers and booleans, never strings. */
export function buildJob(
  task: TaskType,
  datasetVersionId: string,
  values: ParamValues,
): { body?: ClassificationJob | DetectionJob; errors: Record<string, string> } {
  const errors: Record<string, string> = {};
  const hyperparams: Record<string, number | boolean> = {};
  for (const spec of PARAM_SPECS[task]) {
    const raw = values[spec.key];
    if (spec.kind === "bool") {
      hyperparams[spec.key] = raw === true;
      continue;
    }
    const { value, error } = checkNumber(spec, typeof raw === "string" ? raw : "");
    if (error) errors[spec.key] = error;
    else hyperparams[spec.key] = value!;
  }
  if (!datasetVersionId) errors.dataset_version_id = "Choose a dataset version.";
  if (Object.keys(errors).length) return { errors };
  const body =
    task === "classification"
      ? ({
          dataset_version_id: datasetVersionId,
          model_arch: "resnet18",
          hyperparams,
        } as ClassificationJob)
      : ({
          dataset_version_id: datasetVersionId,
          model_arch: "yolov8s",
          hyperparams,
        } as DetectionJob);
  return { body, errors };
}

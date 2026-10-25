"use client";

import { Button } from "@/components/ui/button";
import { inputClass } from "@/components/ui/field";
import type { ClassCreate } from "@/lib/api/types";
import { nextColor } from "./class-colors";

export interface DraftClass extends ClassCreate {
  key: number;
}

let counter = 0;
export function draftClass(used: readonly string[]): DraftClass {
  return { key: ++counter, name: "", color: nextColor(used) };
}

/** Validate new class names locally; existing names are compared exactly, like the API. */
export function classDraftErrors(drafts: DraftClass[], existing: readonly string[] = []) {
  const errors: Record<number, string> = {};
  const seen = new Set(existing);
  for (const draft of drafts) {
    const name = draft.name.trim();
    if (!name) errors[draft.key] = "Enter a class name.";
    else if (name.length > 200) errors[draft.key] = "Use at most 200 characters.";
    else if (seen.has(name)) errors[draft.key] = "This class name is already used.";
    seen.add(name);
  }
  return errors;
}

export function ClassListEditor({
  drafts,
  onChange,
  errors,
  existingColors = [],
}: {
  drafts: DraftClass[];
  onChange: (drafts: DraftClass[]) => void;
  errors: Record<number, string>;
  existingColors?: string[];
}) {
  const update = (key: number, patch: Partial<DraftClass>) =>
    onChange(drafts.map((d) => (d.key === key ? { ...d, ...patch } : d)));
  return (
    <div className="flex flex-col gap-2">
      {drafts.map((draft, index) => (
        <div key={draft.key} className="flex flex-col gap-1">
          <div className="flex items-center gap-2">
            <input
              type="color"
              aria-label={`Color for class ${index + 1}`}
              className="h-9 w-11 cursor-pointer rounded border border-line bg-white p-1"
              value={draft.color}
              onChange={(e) => update(draft.key, { color: e.target.value })}
            />
            <input
              aria-label={`Name of class ${index + 1}`}
              aria-invalid={errors[draft.key] ? true : undefined}
              className={inputClass}
              placeholder="e.g. scratch"
              value={draft.name}
              maxLength={200}
              onChange={(e) => update(draft.key, { name: e.target.value })}
            />
            <Button
              size="sm"
              variant="ghost"
              aria-label={`Remove class ${index + 1}`}
              onClick={() => onChange(drafts.filter((d) => d.key !== draft.key))}
            >
              Remove
            </Button>
          </div>
          {errors[draft.key] && <p className="pl-14 text-sm text-danger">{errors[draft.key]}</p>}
        </div>
      ))}
      <div>
        <Button
          size="sm"
          onClick={() =>
            onChange([...drafts, draftClass([...existingColors, ...drafts.map((d) => d.color)])])
          }
        >
          Add class
        </Button>
      </div>
    </div>
  );
}

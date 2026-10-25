/** Distinct, high-contrast defaults offered for new classes. */
export const CLASS_PALETTE = [
  "#16a34a",
  "#dc2626",
  "#2563eb",
  "#d97706",
  "#9333ea",
  "#0891b2",
  "#db2777",
  "#65a30d",
  "#4f46e5",
  "#ea580c",
];

export function nextColor(used: readonly string[]): string {
  return (
    CLASS_PALETTE.find((c) => !used.includes(c)) ??
    CLASS_PALETTE[used.length % CLASS_PALETTE.length]!
  );
}

export const COLOR_PATTERN = /^#[0-9a-fA-F]{6}$/;

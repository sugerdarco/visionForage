type Tone = "neutral" | "blue" | "green" | "amber" | "red" | "violet";

const TONES: Record<Tone, string> = {
  neutral: "bg-slate-100 text-slate-800 ring-slate-300",
  blue: "bg-blue-50 text-blue-900 ring-blue-200",
  green: "bg-green-50 text-green-900 ring-green-200",
  amber: "bg-amber-50 text-amber-900 ring-amber-300",
  red: "bg-red-50 text-red-900 ring-red-200",
  violet: "bg-violet-50 text-violet-900 ring-violet-200",
};

export function Badge({ tone = "neutral", children }: { tone?: Tone; children: React.ReactNode }) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${TONES[tone]}`}
    >
      {children}
    </span>
  );
}

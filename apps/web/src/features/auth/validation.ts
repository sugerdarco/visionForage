const encoder = new TextEncoder();

/** Mirrors the API: 12+ characters, at most 72 UTF-8 bytes, never trimmed. */
export function passwordError(password: string): string | null {
  if (password.length < 12) return "Use at least 12 characters.";
  if (encoder.encode(password).length > 72)
    return "Use at most 72 bytes (accented letters and symbols can count as several).";
  return null;
}

export function emailError(email: string): string | null {
  if (!email.trim()) return "Enter your email address.";
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim())) return "Enter a valid email address.";
  return null;
}

/** Only same-app absolute paths are accepted as post-login destinations. */
export function safeNext(value: string | null): string {
  return value && value.startsWith("/") && !value.startsWith("//") ? value : "/projects";
}

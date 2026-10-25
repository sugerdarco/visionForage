export interface SseFrame {
  event: string;
  data: string;
  /** The `id:` field sent in this frame, if any. */
  id: string | undefined;
}

/**
 * Incremental parser for the text/event-stream format
 * (https://html.spec.whatwg.org/multipage/server-sent-events.html#parsing-an-event-stream).
 * Feed it decoded text in arbitrary chunks; it returns complete frames only.
 * Decode bytes with `TextDecoder.decode(chunk, {stream: true})` so split UTF-8
 * sequences are carried over by the decoder rather than corrupted here.
 */
export class SseParser {
  private buffer = "";
  private skipLeadingLineFeed = false;
  private eventType = "";
  private dataLines: string[] = [];
  private frameId: string | undefined;
  private started = false;
  /** Latest `retry:` value in milliseconds. */
  retry: number | null = null;

  push(text: string): SseFrame[] {
    if (!this.started) {
      this.started = true;
      if (text.startsWith("﻿")) text = text.slice(1);
    }
    if (this.skipLeadingLineFeed && text.startsWith("\n")) text = text.slice(1);
    this.skipLeadingLineFeed = false;
    this.buffer += text;
    const frames: SseFrame[] = [];
    let start = 0;
    for (let i = 0; i < this.buffer.length; i++) {
      const char = this.buffer[i];
      if (char !== "\n" && char !== "\r") continue;
      const line = this.buffer.slice(start, i);
      if (char === "\r") {
        if (i + 1 === this.buffer.length) this.skipLeadingLineFeed = true;
        else if (this.buffer[i + 1] === "\n") i++;
      }
      start = i + 1;
      const frame = this.line(line);
      if (frame) frames.push(frame);
    }
    this.buffer = this.buffer.slice(start);
    return frames;
  }

  private line(line: string): SseFrame | null {
    if (line === "") return this.dispatch();
    if (line.startsWith(":")) return null;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    switch (field) {
      case "event":
        this.eventType = value;
        break;
      case "data":
        this.dataLines.push(value);
        break;
      case "id":
        if (!value.includes("\0")) this.frameId = value;
        break;
      case "retry":
        if (/^\d+$/.test(value)) this.retry = Number(value);
        break;
    }
    return null;
  }

  private dispatch(): SseFrame | null {
    const frame =
      this.dataLines.length > 0
        ? { event: this.eventType || "message", data: this.dataLines.join("\n"), id: this.frameId }
        : null;
    this.eventType = "";
    this.dataLines = [];
    this.frameId = undefined;
    return frame;
  }
}

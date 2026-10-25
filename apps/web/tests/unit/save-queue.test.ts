import { describe, expect, it } from "vitest";
import { AnnotationSaveQueue } from "@/features/annotations/save-queue";
import { ApiError } from "@/lib/api/errors";
import type { AnnotationSave } from "@/lib/api/types";

function deferred() {
  let resolve!: () => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<void>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

const body = (classId: string): AnnotationSave => ({ annotations: [{ class_id: classId }] });

describe("annotation save ordering", () => {
  it("never sends an older draft after a newer one and coalesces queued edits", async () => {
    const sent: string[] = [];
    const gates: ReturnType<typeof deferred>[] = [];
    const queue = new AnnotationSaveQueue((b) => {
      sent.push(b.annotations[0]!.class_id);
      const gate = deferred();
      gates.push(gate);
      return gate.promise;
    });
    void queue.save(1, body("a"));
    void queue.save(2, body("b"));
    void queue.save(3, body("c"));
    // A stale autosave arriving late is ignored.
    void queue.save(2, body("stale"));
    expect(sent).toEqual(["a"]);
    gates[0]!.resolve();
    await Promise.resolve();
    await Promise.resolve();
    expect(sent).toEqual(["a", "c"]);
    gates[1]!.resolve();
    await queue.idle();
    expect(queue.getSnapshot().state).toBe("saved");
    expect(queue.getSnapshot().savedRevision).toBe(3);
  });

  it("keeps the draft dirty after a failure and retries only the latest body", async () => {
    let fail = true;
    const sent: string[] = [];
    const queue = new AnnotationSaveQueue(async (b) => {
      sent.push(b.annotations[0]!.class_id);
      if (fail) throw new ApiError({ kind: "unavailable", status: 503, message: "down" });
    });
    await queue.save(1, body("a"));
    expect(queue.getSnapshot().state).toBe("error");
    expect(queue.isDirty).toBe(true);
    expect(queue.getSnapshot().error?.message).toBe("down");
    fail = false;
    await queue.retry();
    expect(sent).toEqual(["a", "a"]);
    expect(queue.getSnapshot().state).toBe("saved");
    expect(queue.isDirty).toBe(false);
  });

  it("lets a newer successful save supersede an older failure", async () => {
    const gates: ReturnType<typeof deferred>[] = [];
    const queue = new AnnotationSaveQueue(() => {
      const gate = deferred();
      gates.push(gate);
      return gate.promise;
    });
    void queue.save(1, body("a"));
    void queue.save(2, body("b"));
    gates[0]!.reject(new Error("lost"));
    await new Promise((r) => setTimeout(r, 0));
    expect(queue.getSnapshot().state).toBe("saving");
    gates[1]!.resolve();
    await queue.idle();
    expect(queue.getSnapshot()).toMatchObject({ state: "saved", error: null, savedRevision: 2 });
  });
});

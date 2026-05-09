import type { SessionSummary } from "../types";

// ─── Time formatting ──────────────────────────────────────────────────────────

export function formatMs(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

export function formatSessionTime(ts: number): string {
  const diff = Date.now() - ts;
  const hour = new Date(ts).getHours();
  const ampm = hour >= 12 ? "pm" : "am";
  const h12 = hour % 12 || 12;

  if (diff < 60 * 60 * 1000) return "Just now";
  if (diff < 24 * 60 * 60 * 1000) return `Today, ${h12}${ampm}`;
  if (diff < 48 * 60 * 60 * 1000) return `Yesterday, ${h12}${ampm}`;
  return new Date(ts).toLocaleDateString(undefined, { weekday: "short", hour: "numeric" });
}

// ─── Context snapshot ────────────────────────────────────────────────────────

export function currentGreeting(): string {
  const h = new Date().getHours();
  if (h < 5) return "Late night";
  if (h < 12) return "Morning";
  if (h < 17) return "Afternoon";
  if (h < 21) return "Evening";
  return "Night";
}

export function currentHourLabel(): string {
  const now = new Date();
  return `${now.getHours()}:${String(now.getMinutes()).padStart(2, "0")}`;
}

// ─── Session grouping ────────────────────────────────────────────────────────

const SESSION_WINDOW_MS = 2 * 60 * 60 * 1000; // 2 hours

export function groupEventsIntoSessions(
  events: { clientTimestampMs: number; skip: boolean }[]
): SessionSummary[] {
  if (events.length === 0) return [];

  const sorted = [...events].sort((a, b) => b.clientTimestampMs - a.clientTimestampMs);
  const sessions: SessionSummary[] = [];
  let current: SessionSummary = { tracks: 0, skips: 0, ts: sorted[0].clientTimestampMs };

  for (const ev of sorted) {
    if (current.ts - ev.clientTimestampMs > SESSION_WINDOW_MS) {
      sessions.push(current);
      current = { tracks: 0, skips: 0, ts: ev.clientTimestampMs };
    }
    current.tracks++;
    if (ev.skip) current.skips++;
  }
  sessions.push(current);
  return sessions;
}

// ─── UUID generation ──────────────────────────────────────────────────────────

export function generateId(): string {
  // RFC 4122 v4 UUID using Math.random (good enough for client-side dedup keys)
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
  });
}

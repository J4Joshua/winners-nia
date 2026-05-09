import type { LocationBucket, ContextSnapshot } from "../types";

export type { LocationBucket, ContextSnapshot };

export function buildContextSnapshot(overrides?: Partial<ContextSnapshot>): ContextSnapshot {
  const now = new Date();
  return {
    hourLocal: now.getHours(),
    dow: now.getDay() === 0 ? 6 : now.getDay() - 1, // Mon=0, Sun=6
    locationBucket: "other" as LocationBucket,
    sessionSkipRate: 0,
    consecutiveSkips: 0,
    ...overrides,
  };
}

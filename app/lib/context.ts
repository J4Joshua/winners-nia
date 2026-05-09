export type LocationBucket = "home" | "gym" | "transit" | "other";

export interface ContextSnapshot {
  hourLocal: number;
  dow: number;
  locationBucket: LocationBucket;
  sessionSkipRate: number;
  consecutiveSkips: number;
}

export function buildContextSnapshot(overrides?: Partial<ContextSnapshot>): ContextSnapshot {
  const now = new Date();
  return {
    hourLocal: now.getHours(),
    dow: now.getDay() === 0 ? 6 : now.getDay() - 1, // Mon=0
    locationBucket: "other",
    sessionSkipRate: 0,
    consecutiveSkips: 0,
    ...overrides,
  };
}

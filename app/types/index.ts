// ─── Domain enums ─────────────────────────────────────────────────────────────

export type LocationBucket = "home" | "gym" | "transit" | "other";
export type SkipBucket = "instant" | "early" | "late" | "none";
export type WeatherCondition = "clear" | "clouds" | "rain" | "snow" | "storm";

// ─── Entities ─────────────────────────────────────────────────────────────────

export interface Track {
  uri: string;
  name: string;
  artistName: string;
  albumArtUri?: string;
  durationMs: number;
}

export interface WeatherData {
  temp: number;
  condition: WeatherCondition;
  humidity: number;
  isDay: boolean;
}

export interface ContextSnapshot {
  hourLocal: number;
  dow: number;
  locationBucket: LocationBucket;
  sessionSkipRate: number;
  consecutiveSkips: number;
}

// ─── Telemetry ────────────────────────────────────────────────────────────────

export interface PlaybackEvent {
  eventId: string;
  spotifyTrackUri: string;
  listenRatio: number;
  skip: boolean;
  skipBucket: SkipBucket;
  skipPositionMs?: number;
  replay: boolean;
  positionMs?: number;
  durationMs?: number;
  volumeDelta?: number;
  hourLocal: number;
  dow: number;
  locationBucket: LocationBucket;
  sessionSkipRate: number;
  consecutiveSkips: number;
  sessionIdx: number;
  weather?: WeatherData;
  clientSchemaVersion: string;
  clientTimestampMs: number;
}

// ─── Session summary (derived from events) ────────────────────────────────────

export interface SessionSummary {
  tracks: number;
  skips: number;
  ts: number;
}

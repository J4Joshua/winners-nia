import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

export default defineSchema({
  users: defineTable({
    spotifyId: v.string(),
    displayName: v.string(),
    email: v.optional(v.string()),
    avatarUrl: v.optional(v.string()),
    // Stored server-side; never exposed to client
    spotifyRefreshToken: v.string(),
    spotifyAccessToken: v.string(),
    spotifyTokenExpiresAt: v.number(),
    onboardingStatus: v.union(
      v.literal("pending"),
      v.literal("fetching"),
      v.literal("training"),
      v.literal("ready")
    ),
    weatherEnabled: v.boolean(),
    locationBucketOverride: v.optional(
      v.union(
        v.literal("home"),
        v.literal("gym"),
        v.literal("transit"),
        v.literal("other")
      )
    ),
    createdAt: v.number(),
    updatedAt: v.number(),
  })
    .index("by_spotify_id", ["spotifyId"])
    .index("by_onboarding_status", ["onboardingStatus"]),

  playbackEvents: defineTable({
    userId: v.id("users"),
    eventId: v.string(),
    spotifyTrackUri: v.string(),
    listenRatio: v.number(),
    skip: v.boolean(),
    skipBucket: v.union(
      v.literal("instant"),
      v.literal("early"),
      v.literal("late"),
      v.literal("none")
    ),
    skipPositionMs: v.optional(v.number()),
    replay: v.boolean(),
    positionMs: v.optional(v.number()),
    durationMs: v.optional(v.number()),
    volumeDelta: v.optional(v.number()),
    hourLocal: v.number(),
    dow: v.number(),
    locationBucket: v.union(
      v.literal("home"),
      v.literal("gym"),
      v.literal("transit"),
      v.literal("other")
    ),
    sessionSkipRate: v.number(),
    consecutiveSkips: v.number(),
    sessionIdx: v.number(),
    weather: v.optional(
      v.object({
        temp: v.number(),
        condition: v.union(
          v.literal("clear"),
          v.literal("clouds"),
          v.literal("rain"),
          v.literal("snow"),
          v.literal("storm")
        ),
        humidity: v.number(),
        isDay: v.boolean(),
      })
    ),
    clientSchemaVersion: v.string(),
    clientTimestampMs: v.number(),
  })
    .index("by_user", ["userId"])
    .index("by_user_and_time", ["userId", "clientTimestampMs"])
    .index("by_event_id", ["eventId"]),

  audioFeatures: defineTable({
    spotifyTrackId: v.string(),
    danceability: v.number(),
    energy: v.number(),
    speechiness: v.number(),
    acousticness: v.number(),
    instrumentalness: v.number(),
    liveness: v.number(),
    valence: v.number(),
    loudness: v.number(),
    tempo: v.number(),
    mode: v.number(),
    key: v.number(),
    explicit: v.boolean(),
    popularity: v.number(),
    durationMs: v.number(),
    timeSignature: v.number(),
    fetchedAt: v.number(),
  }).index("by_track_id", ["spotifyTrackId"]),

  userWeights: defineTable({
    userId: v.id("users"),
    weightsBase64: v.string(),
    modelVersion: v.string(),
    trainedAt: v.number(),
    eventCount: v.number(),
  })
    .index("by_user", ["userId"])
    .index("by_user_and_version", ["userId", "modelVersion"]),

  onboardingJobs: defineTable({
    userId: v.id("users"),
    status: v.union(
      v.literal("fetching"),
      v.literal("analyzing"),
      v.literal("training"),
      v.literal("ready"),
      v.literal("error")
    ),
    errorMessage: v.optional(v.string()),
    tracksFound: v.optional(v.number()),
    startedAt: v.number(),
    completedAt: v.optional(v.number()),
  })
    .index("by_user", ["userId"])
    .index("by_status", ["status"]),

  sessions: defineTable({
    userId: v.id("users"),
    token: v.string(),
    createdAt: v.number(),
  }).index("by_token", ["token"]),

  playlistRequests: defineTable({
    userId: v.id("users"),
    contextSnapshot: v.object({
      hourLocal: v.number(),
      dow: v.number(),
      locationBucket: v.union(
        v.literal("home"),
        v.literal("gym"),
        v.literal("transit"),
        v.literal("other")
      ),
      sessionSkipRate: v.number(),
      consecutiveSkips: v.number(),
    }),
    trackUris: v.array(v.string()),
    requestedAt: v.number(),
  }).index("by_user", ["userId"]),
});

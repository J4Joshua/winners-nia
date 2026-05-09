import { mutation } from "./_generated/server";
import { v } from "convex/values";
import { requireSession } from "./lib/auth";

const weatherValidator = v.optional(
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
);

const eventValidator = v.object({
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
  weather: weatherValidator,
  clientSchemaVersion: v.string(),
  clientTimestampMs: v.number(),
});

export const appendBatch = mutation({
  args: {
    sessionToken: v.string(),
    events: v.array(eventValidator),
  },
  handler: async (ctx, { sessionToken, events }) => {
    const user = await requireSession(ctx, sessionToken);

    let inserted = 0;
    let skipped = 0;

    for (const event of events) {
      const existing = await ctx.db
        .query("playbackEvents")
        .withIndex("by_event_id", (q) => q.eq("eventId", event.eventId))
        .unique();

      if (existing) {
        skipped++;
        continue;
      }

      await ctx.db.insert("playbackEvents", { userId: user._id, ...event });
      inserted++;
    }

    return { inserted, skipped };
  },
});

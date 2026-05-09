import { mutation } from "./_generated/server";
import { v } from "convex/values";

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
    userId: v.id("users"),
    events: v.array(eventValidator),
  },
  handler: async (ctx, { userId, events }) => {
    const user = await ctx.db.get(userId);
    if (!user) throw new Error("User not found");

    const inserted: string[] = [];

    for (const event of events) {
      const existing = await ctx.db
        .query("playbackEvents")
        .withIndex("by_event_id", (q) => q.eq("eventId", event.eventId))
        .unique();

      if (existing) continue;

      await ctx.db.insert("playbackEvents", { userId, ...event });
      inserted.push(event.eventId);
    }

    return { inserted: inserted.length, skipped: events.length - inserted.length };
  },
});

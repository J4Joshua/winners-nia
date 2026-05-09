import { query } from "./_generated/server";
import { v } from "convex/values";
import { requireSession } from "./lib/auth";

export const getMe = query({
  args: { sessionToken: v.string() },
  handler: async (ctx, { sessionToken }) => {
    const user = await requireSession(ctx, sessionToken);
    return {
      _id: user._id,
      spotifyId: user.spotifyId,
      displayName: user.displayName,
      email: user.email,
      avatarUrl: user.avatarUrl,
      onboardingStatus: user.onboardingStatus,
      weatherEnabled: user.weatherEnabled,
      locationBucketOverride: user.locationBucketOverride,
    };
  },
});

export const getOnboardingJob = query({
  args: { sessionToken: v.string() },
  handler: async (ctx, { sessionToken }) => {
    const user = await requireSession(ctx, sessionToken);
    return await ctx.db
      .query("onboardingJobs")
      .withIndex("by_user", (q) => q.eq("userId", user._id))
      .order("desc")
      .first();
  },
});

export const getRecentEvents = query({
  args: { sessionToken: v.string(), limit: v.optional(v.number()) },
  handler: async (ctx, { sessionToken, limit = 50 }) => {
    const user = await requireSession(ctx, sessionToken);
    return await ctx.db
      .query("playbackEvents")
      .withIndex("by_user_and_time", (q) => q.eq("userId", user._id))
      .order("desc")
      .take(limit);
  },
});

export const getUserWeights = query({
  args: { sessionToken: v.string() },
  handler: async (ctx, { sessionToken }) => {
    const user = await requireSession(ctx, sessionToken);
    return await ctx.db
      .query("userWeights")
      .withIndex("by_user", (q) => q.eq("userId", user._id))
      .order("desc")
      .first();
  },
});

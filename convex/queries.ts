import { query } from "./_generated/server";
import { v } from "convex/values";

export const getUser = query({
  args: { userId: v.id("users") },
  handler: async (ctx, { userId }) => {
    const user = await ctx.db.get(userId);
    if (!user) return null;

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
  args: { userId: v.id("users") },
  handler: async (ctx, { userId }) => {
    return await ctx.db
      .query("onboardingJobs")
      .withIndex("by_user", (q) => q.eq("userId", userId))
      .order("desc")
      .first();
  },
});

export const getRecentEvents = query({
  args: { userId: v.id("users"), limit: v.optional(v.number()) },
  handler: async (ctx, { userId, limit = 50 }) => {
    return await ctx.db
      .query("playbackEvents")
      .withIndex("by_user_and_time", (q) => q.eq("userId", userId))
      .order("desc")
      .take(limit);
  },
});

export const getUserWeights = query({
  args: { userId: v.id("users") },
  handler: async (ctx, { userId }) => {
    return await ctx.db
      .query("userWeights")
      .withIndex("by_user", (q) => q.eq("userId", userId))
      .order("desc")
      .first();
  },
});

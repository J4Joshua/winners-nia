import { internalMutation, internalQuery } from "./_generated/server";
import { v } from "convex/values";

export const upsertUser = internalMutation({
  args: {
    spotifyId: v.string(),
    displayName: v.string(),
    email: v.optional(v.string()),
    avatarUrl: v.optional(v.string()),
    spotifyAccessToken: v.string(),
    spotifyRefreshToken: v.string(),
    spotifyTokenExpiresAt: v.number(),
  },
  handler: async (ctx, args) => {
    const existing = await ctx.db
      .query("users")
      .withIndex("by_spotify_id", (q) => q.eq("spotifyId", args.spotifyId))
      .unique();

    const now = Date.now();

    if (existing) {
      await ctx.db.patch(existing._id, {
        displayName: args.displayName,
        email: args.email,
        avatarUrl: args.avatarUrl,
        spotifyAccessToken: args.spotifyAccessToken,
        spotifyRefreshToken: args.spotifyRefreshToken,
        spotifyTokenExpiresAt: args.spotifyTokenExpiresAt,
        updatedAt: now,
      });
      return existing._id;
    }

    return await ctx.db.insert("users", {
      ...args,
      onboardingStatus: "pending",
      weatherEnabled: false,
      createdAt: now,
      updatedAt: now,
    });
  },
});

export const createSession = internalMutation({
  args: {
    userId: v.id("users"),
    token: v.string(),
  },
  handler: async (ctx, { userId, token }) => {
    return await ctx.db.insert("sessions", {
      userId,
      token,
      createdAt: Date.now(),
    });
  },
});

export const deleteSession = internalMutation({
  args: { token: v.string() },
  handler: async (ctx, { token }) => {
    const session = await ctx.db
      .query("sessions")
      .withIndex("by_token", (q) => q.eq("token", token))
      .unique();
    if (session) await ctx.db.delete(session._id);
  },
});

export const getBySpotifyId = internalQuery({
  args: { spotifyId: v.string() },
  handler: async (ctx, { spotifyId }) => {
    return await ctx.db
      .query("users")
      .withIndex("by_spotify_id", (q) => q.eq("spotifyId", spotifyId))
      .unique();
  },
});

export const getById = internalQuery({
  args: { userId: v.id("users") },
  handler: async (ctx, { userId }) => ctx.db.get(userId),
});

export const getSessionByToken = internalQuery({
  args: { token: v.string() },
  handler: async (ctx, { token }) => {
    return await ctx.db
      .query("sessions")
      .withIndex("by_token", (q) => q.eq("token", token))
      .unique();
  },
});

export const refreshAccessToken = internalMutation({
  args: { userId: v.id("users"), accessToken: v.string(), refreshToken: v.string(), expiresAt: v.number() },
  handler: async (ctx, { userId, accessToken, refreshToken, expiresAt }) => {
    await ctx.db.patch(userId, {
      spotifyAccessToken: accessToken,
      spotifyRefreshToken: refreshToken,
      spotifyTokenExpiresAt: expiresAt,
      updatedAt: Date.now(),
    });
  },
});

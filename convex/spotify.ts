"use node";

import { action, internalMutation, internalQuery } from "./_generated/server";
import { internal } from "./_generated/api";
import { v } from "convex/values";

const SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token";

export const link = action({
  args: {
    code: v.string(),
    redirectUri: v.string(),
    codeVerifier: v.string(),
  },
  handler: async (ctx, { code, redirectUri, codeVerifier }) => {
    const clientId = process.env.SPOTIFY_CLIENT_ID;
    if (!clientId) throw new Error("SPOTIFY_CLIENT_ID not configured");

    const body = new URLSearchParams({
      grant_type: "authorization_code",
      code,
      redirect_uri: redirectUri,
      client_id: clientId,
      code_verifier: codeVerifier,
    });

    const tokenRes = await fetch(SPOTIFY_TOKEN_URL, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: body.toString(),
    });

    if (!tokenRes.ok) {
      const err = await tokenRes.text();
      throw new Error(`Spotify token exchange failed: ${err}`);
    }

    const tokens = (await tokenRes.json()) as {
      access_token: string;
      refresh_token: string;
      expires_in: number;
    };

    const profileRes = await fetch("https://api.spotify.com/v1/me", {
      headers: { Authorization: `Bearer ${tokens.access_token}` },
    });

    if (!profileRes.ok) throw new Error("Failed to fetch Spotify profile");

    const profile = (await profileRes.json()) as {
      id: string;
      display_name: string;
      email?: string;
      images?: { url: string }[];
    };

    const userId = await ctx.runMutation(internal.spotify.upsertUser, {
      spotifyId: profile.id,
      displayName: profile.display_name ?? profile.id,
      email: profile.email,
      avatarUrl: profile.images?.[0]?.url,
      spotifyAccessToken: tokens.access_token,
      spotifyRefreshToken: tokens.refresh_token,
      spotifyTokenExpiresAt: Date.now() + tokens.expires_in * 1000,
    });

    return { userId };
  },
});

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

export const refreshAccessToken = internalMutation({
  args: { userId: v.id("users") },
  handler: async (ctx, { userId }) => {
    const user = await ctx.db.get(userId);
    if (!user) throw new Error("User not found");

    const clientId = process.env.SPOTIFY_CLIENT_ID;
    if (!clientId) throw new Error("SPOTIFY_CLIENT_ID not configured");

    const body = new URLSearchParams({
      grant_type: "refresh_token",
      refresh_token: user.spotifyRefreshToken,
      client_id: clientId,
    });

    const res = await fetch(SPOTIFY_TOKEN_URL, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: body.toString(),
    });

    if (!res.ok) throw new Error("Failed to refresh Spotify token");

    const tokens = (await res.json()) as {
      access_token: string;
      refresh_token?: string;
      expires_in: number;
    };

    await ctx.db.patch(userId, {
      spotifyAccessToken: tokens.access_token,
      spotifyRefreshToken: tokens.refresh_token ?? user.spotifyRefreshToken,
      spotifyTokenExpiresAt: Date.now() + tokens.expires_in * 1000,
      updatedAt: Date.now(),
    });

    return tokens.access_token;
  },
});

export const getMe = internalQuery({
  args: { userId: v.id("users") },
  handler: async (ctx, { userId }) => ctx.db.get(userId),
});

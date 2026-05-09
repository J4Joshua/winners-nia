"use node";

import { action } from "./_generated/server";
import { internal } from "./_generated/api";
import { v } from "convex/values";
import crypto from "node:crypto";

const SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token";

function generateSessionToken(): string {
  return crypto.randomBytes(32).toString("hex");
}

async function spotifyPost(
  url: string,
  params: Record<string, string>
): Promise<Response> {
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams(params).toString(),
  });
}

// ─── Public actions ───────────────────────────────────────────────────────────

export const link = action({
  args: {
    code: v.string(),
    redirectUri: v.string(),
    codeVerifier: v.string(),
  },
  handler: async (ctx, { code, redirectUri, codeVerifier }) => {
    const clientId = process.env.SPOTIFY_CLIENT_ID;
    if (!clientId) throw new Error("SPOTIFY_CLIENT_ID not configured");

    const tokenRes = await spotifyPost(SPOTIFY_TOKEN_URL, {
      grant_type: "authorization_code",
      code,
      redirect_uri: redirectUri,
      client_id: clientId,
      code_verifier: codeVerifier,
    });

    if (!tokenRes.ok) {
      const body = await tokenRes.text();
      throw new Error(`Spotify token exchange failed (${tokenRes.status}): ${body}`);
    }

    const tokens = (await tokenRes.json()) as {
      access_token: string;
      refresh_token: string;
      expires_in: number;
    };

    if (!tokens.access_token || !tokens.refresh_token) {
      throw new Error("Spotify returned an incomplete token response");
    }

    const profileRes = await fetch("https://api.spotify.com/v1/me", {
      headers: { Authorization: `Bearer ${tokens.access_token}` },
    });

    if (!profileRes.ok) {
      throw new Error(`Failed to fetch Spotify profile (${profileRes.status})`);
    }

    const profile = (await profileRes.json()) as {
      id: string;
      display_name?: string;
      email?: string;
      images?: { url: string }[];
    };

    const userId = await ctx.runMutation(internal.users.upsertUser, {
      spotifyId: profile.id,
      displayName: profile.display_name ?? profile.id,
      email: profile.email,
      avatarUrl: profile.images?.[0]?.url,
      spotifyAccessToken: tokens.access_token,
      spotifyRefreshToken: tokens.refresh_token,
      spotifyTokenExpiresAt: Date.now() + tokens.expires_in * 1000,
    });

    const sessionToken = generateSessionToken();
    await ctx.runMutation(internal.users.createSession, { userId, token: sessionToken });

    return { sessionToken };
  },
});

export const refreshSpotifyToken = action({
  args: { sessionToken: v.string() },
  handler: async (ctx, { sessionToken }) => {
    const clientId = process.env.SPOTIFY_CLIENT_ID;
    if (!clientId) throw new Error("SPOTIFY_CLIENT_ID not configured");

    const session = await ctx.runQuery(internal.users.getSessionByToken, { token: sessionToken });
    if (!session) throw new Error("Unauthorized");

    const user = await ctx.runQuery(internal.users.getById, { userId: session.userId });
    if (!user) throw new Error("User not found");

    const res = await spotifyPost(SPOTIFY_TOKEN_URL, {
      grant_type: "refresh_token",
      refresh_token: user.spotifyRefreshToken,
      client_id: clientId,
    });

    if (!res.ok) {
      const body = await res.text();
      throw new Error(`Token refresh failed (${res.status}): ${body}`);
    }

    const tokens = (await res.json()) as {
      access_token: string;
      refresh_token?: string;
      expires_in: number;
    };

    await ctx.runMutation(internal.users.refreshAccessToken, {
      userId: session.userId,
      accessToken: tokens.access_token,
      refreshToken: tokens.refresh_token ?? user.spotifyRefreshToken,
      expiresAt: Date.now() + tokens.expires_in * 1000,
    });

    return { accessToken: tokens.access_token };
  },
});

export const logout = action({
  args: { sessionToken: v.string() },
  handler: async (ctx, { sessionToken }) => {
    await ctx.runMutation(internal.users.deleteSession, { token: sessionToken });
  },
});

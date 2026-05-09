import { useCallback, useState } from "react";
import { useAction } from "convex/react";
import { api } from "../convex/_generated/api";
import type { Id } from "../convex/_generated/dataModel";
import { useSpotifyAuthRequest } from "../lib/spotify";
import * as AuthSession from "expo-auth-session";
import AsyncStorage from "@react-native-async-storage/async-storage";

const USER_ID_KEY = "@attune/userId";

export function useSpotifyAuth() {
  const [userId, setUserId] = useState<Id<"users"> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { request, promptAsync, redirectUri } = useSpotifyAuthRequest();
  const linkSpotify = useAction(api.spotify.link);

  const loadStoredUserId = useCallback(async () => {
    const stored = await AsyncStorage.getItem(USER_ID_KEY);
    if (stored) setUserId(stored as Id<"users">);
    return stored as Id<"users"> | null;
  }, []);

  const login = useCallback(async () => {
    if (!request) return;
    setLoading(true);
    setError(null);

    try {
      const result = await promptAsync();

      if (result.type !== "success") {
        setError(result.type === "cancel" ? null : "Authentication failed");
        return;
      }

      const { code } = result.params;
      const codeVerifier = request.codeVerifier;

      if (!codeVerifier) throw new Error("Missing code verifier");

      const { userId: newUserId } = await linkSpotify({
        code,
        redirectUri,
        codeVerifier,
      });

      await AsyncStorage.setItem(USER_ID_KEY, newUserId);
      setUserId(newUserId as Id<"users">);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Login failed");
    } finally {
      setLoading(false);
    }
  }, [request, promptAsync, redirectUri, linkSpotify]);

  const logout = useCallback(async () => {
    await AsyncStorage.removeItem(USER_ID_KEY);
    setUserId(null);
  }, []);

  return { userId, loading, error, login, logout, loadStoredUserId };
}

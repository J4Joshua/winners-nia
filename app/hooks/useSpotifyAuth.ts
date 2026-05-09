import { useCallback } from "react";
import { useAction, useMutation } from "convex/react";
import { api } from "../convex/_generated/api";
import { useSpotifyAuthRequest } from "../lib/spotify";

export function useSpotifyAuth(setSession: (token: string | null) => Promise<void>) {
  const { request, promptAsync, redirectUri } = useSpotifyAuthRequest();
  const linkSpotify = useAction(api.spotify.link);
  const startOnboarding = useMutation(api.onboarding.startOnboarding);

  const login = useCallback(async () => {
    if (!request) throw new Error("Auth request not ready");

    const result = await promptAsync();
    if (result.type === "cancel") return null;
    if (result.type !== "success") throw new Error("Authentication failed");

    const codeVerifier = request.codeVerifier;
    if (!codeVerifier) throw new Error("Missing PKCE code verifier");

    const { sessionToken } = await linkSpotify({
      code: result.params.code,
      redirectUri,
      codeVerifier,
    });

    await setSession(sessionToken);

    // Kick off onboarding — errors are surfaced via the onboarding job query
    await startOnboarding({ sessionToken }).catch((err) => {
      console.warn("[useSpotifyAuth] startOnboarding failed:", err);
    });

    return sessionToken;
  }, [request, promptAsync, redirectUri, linkSpotify, setSession, startOnboarding]);

  return { login, isReady: !!request };
}

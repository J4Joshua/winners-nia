import { useCallback, useEffect, useRef } from "react";
import { useMutation } from "convex/react";
import { AppState } from "react-native";
import { api } from "../convex/_generated/api";
import type { PlaybackEvent } from "../types";
export type { PlaybackEvent };

const FLUSH_THRESHOLD = 5;
const FLUSH_INTERVAL_MS = 30_000;

export function useTelemetry(sessionToken: string | null) {
  const queueRef = useRef<PlaybackEvent[]>([]);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const appendBatch = useMutation(api.events.appendBatch);

  const clearTimer = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const flush = useCallback(async () => {
    if (!sessionToken || queueRef.current.length === 0) return;
    const batch = [...queueRef.current];
    queueRef.current = [];
    clearTimer();
    try {
      await appendBatch({ sessionToken, events: batch });
    } catch {
      // Re-queue on failure so events are not lost on transient errors
      queueRef.current = [...batch, ...queueRef.current];
    }
  }, [sessionToken, appendBatch, clearTimer]);

  const track = useCallback(
    (event: PlaybackEvent) => {
      queueRef.current.push(event);

      if (queueRef.current.length >= FLUSH_THRESHOLD) {
        // Clear any pending timer before flushing immediately
        clearTimer();
        void flush();
      } else if (!timerRef.current) {
        timerRef.current = setTimeout(() => {
          timerRef.current = null;
          void flush();
        }, FLUSH_INTERVAL_MS);
      }
    },
    [flush, clearTimer]
  );

  // Flush when the app goes to background
  useEffect(() => {
    const sub = AppState.addEventListener("change", (state) => {
      if (state === "background" || state === "inactive") void flush();
    });
    return () => sub.remove();
  }, [flush]);

  // Flush + clear timer on unmount
  useEffect(() => {
    return () => {
      clearTimer();
      void flush();
    };
  }, [flush, clearTimer]);

  return { track, flush };
}

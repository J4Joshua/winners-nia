import React, { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import type { Track } from "../types";
export type { Track };

interface PlaybackState {
  track: Track | null;
  isPlaying: boolean;
  positionMs: number;
}

interface PlaybackContextValue extends PlaybackState {
  setTrack: (track: Track | null) => void;
  play: () => void;
  pause: () => void;
  toggle: () => void;
  seekTo: (ms: number) => void;
}

const PlaybackContext = createContext<PlaybackContextValue | null>(null);

export function PlaybackProvider({ children }: { children: React.ReactNode }) {
  const [track, setTrackState] = useState<Track | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [positionMs, setPositionMs] = useState(0);
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopTick = useCallback(() => {
    if (tickRef.current !== null) {
      clearInterval(tickRef.current);
      tickRef.current = null;
    }
  }, []);

  // Drive position forward while playing
  useEffect(() => {
    if (isPlaying && track) {
      tickRef.current = setInterval(() => {
        setPositionMs((p) => {
          if (p >= track.durationMs) {
            stopTick();
            return track.durationMs;
          }
          return p + 1000;
        });
      }, 1000);
    } else {
      stopTick();
    }
    return stopTick;
  }, [isPlaying, track, stopTick]);

  // Reset position and auto-play when track changes
  const setTrack = useCallback((next: Track | null) => {
    setPositionMs(0);
    setIsPlaying(next !== null);
    setTrackState(next);
  }, []);

  const play = useCallback(() => setIsPlaying(true), []);
  const pause = useCallback(() => setIsPlaying(false), []);
  const toggle = useCallback(() => setIsPlaying((v) => !v), []);
  const seekTo = useCallback((ms: number) => setPositionMs(ms), []);

  return (
    <PlaybackContext.Provider
      value={{ track, isPlaying, positionMs, setTrack, play, pause, toggle, seekTo }}
    >
      {children}
    </PlaybackContext.Provider>
  );
}

export function usePlayback() {
  const ctx = useContext(PlaybackContext);
  if (!ctx) throw new Error("usePlayback must be used within PlaybackProvider");
  return ctx;
}

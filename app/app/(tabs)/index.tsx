import { useCallback, useMemo, useState } from "react";
import { ScrollView, View, Text, Pressable } from "../../src/tw";
import { AttuneButton } from "../../components/AttuneButton";
import { NowPlayingStrip } from "../../components/NowPlayingStrip";
import { Skeleton, SkeletonText } from "../../components/Skeleton";
import { useAuth } from "../../contexts/AuthContext";
import { usePlayback } from "../../contexts/PlaybackContext";
import { useQuery } from "convex/react";
import { api } from "../../convex/_generated/api";
import { buildContextSnapshot } from "../../lib/context";
import {
  currentGreeting,
  currentHourLabel,
  groupEventsIntoSessions,
  formatSessionTime,
} from "../../lib/utils";
import Animated, { FadeIn, FadeInDown, FadeInUp } from "react-native-reanimated";
import * as Haptics from "expo-haptics";
import { Platform } from "react-native";

const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] as const;
const LOCATION_ICONS: Record<string, string> = {
  home: "🏠",
  gym: "💪",
  transit: "🚇",
  other: "📍",
};

function ContextChip({
  icon,
  label,
  onPress,
}: {
  icon: string;
  label: string;
  onPress?: () => void;
}) {
  return (
    <Pressable
      onPress={onPress}
      className="flex-row items-center gap-1.5 px-3 py-1.5 rounded-full"
      style={{
        backgroundColor: "rgba(255,255,255,0.07)",
        borderWidth: 1,
        borderColor: "rgba(255,255,255,0.1)",
      }}
    >
      <Text className="text-xs">{icon}</Text>
      <Text className="text-text-2 text-xs font-medium">{label}</Text>
    </Pressable>
  );
}

function SessionCard({
  label,
  tracks,
  skipPct,
  delay,
}: {
  label: string;
  tracks: number;
  skipPct: number;
  delay: number;
}) {
  const quality =
    skipPct < 10
      ? { label: "Great session", color: "#00e87a" }
      : skipPct < 25
      ? { label: "Good session", color: "#ffd60a" }
      : { label: "Learning…", color: "#ff9f0a" };

  return (
    <Animated.View entering={FadeInUp.delay(delay).springify().damping(18)}>
      <View
        className="p-4 rounded-2xl gap-3"
        style={{
          backgroundColor: "rgba(255,255,255,0.045)",
          borderWidth: 1,
          borderColor: "rgba(255,255,255,0.07)",
          borderCurve: "continuous",
        }}
      >
        <View className="flex-row items-center justify-between">
          <Text className="text-text-1 text-sm font-semibold">{label}</Text>
          <View
            className="px-2 py-0.5 rounded-full"
            style={{ backgroundColor: `${quality.color}18` }}
          >
            <Text className="text-[11px] font-semibold" style={{ color: quality.color }}>
              {quality.label}
            </Text>
          </View>
        </View>

        <View className="flex-row gap-4">
          <View className="gap-0.5">
            <Text className="text-text-3 text-[10px] uppercase tracking-wider">Tracks</Text>
            <Text
              className="text-text-1 font-bold text-lg"
              style={{ fontVariant: ["tabular-nums"] }}
            >
              {tracks}
            </Text>
          </View>
          <View className="gap-0.5">
            <Text className="text-text-3 text-[10px] uppercase tracking-wider">Skip rate</Text>
            <Text
              className="text-text-1 font-bold text-lg"
              style={{ fontVariant: ["tabular-nums"] }}
            >
              {skipPct}%
            </Text>
          </View>
        </View>

        <View className="h-1 rounded-full bg-surface-3 overflow-hidden">
          <View
            className="h-full rounded-full"
            style={{ width: `${skipPct}%`, backgroundColor: quality.color }}
          />
        </View>
      </View>
    </Animated.View>
  );
}

function EmptyState() {
  return (
    <Animated.View entering={FadeInUp.delay(380).springify().damping(18)}>
      <View
        className="p-6 rounded-2xl items-center gap-3"
        style={{
          backgroundColor: "rgba(255,255,255,0.04)",
          borderWidth: 1,
          borderColor: "rgba(255,255,255,0.07)",
          borderCurve: "continuous",
        }}
      >
        <Text className="text-3xl">🎧</Text>
        <View className="items-center gap-1">
          <Text className="text-text-1 text-sm font-semibold">No sessions yet</Text>
          <Text className="text-text-3 text-xs text-center">
            Tap Attune this moment to start{"\n"}your first session
          </Text>
        </View>
      </View>
    </Animated.View>
  );
}

export default function HomeScreen() {
  const { sessionToken } = useAuth();
  const { track, isPlaying, toggle } = usePlayback();
  const [isAttuning, setIsAttuning] = useState(false);

  const ctx = buildContextSnapshot();
  const hourLabel = currentHourLabel();
  const dayLabel = DAYS[ctx.dow];

  const user = useQuery(api.queries.getMe, sessionToken ? { sessionToken } : "skip");
  const recentEvents = useQuery(
    api.queries.getRecentEvents,
    sessionToken ? { sessionToken, limit: 100 } : "skip"
  );

  const sessionSummaries = useMemo(
    () => (recentEvents ? groupEventsIntoSessions(recentEvents).slice(0, 3) : []),
    [recentEvents]
  );

  const handleAttune = useCallback(async () => {
    if (Platform.OS === "ios") {
      await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
    }
    setIsAttuning(true);
    // TODO: call generatePlaylist action → enqueue to Spotify
    await new Promise<void>((r) => setTimeout(r, 2200));
    setIsAttuning(false);
  }, []);

  const isLoading = user === undefined;

  return (
    <View className="flex-1 bg-surface-0">
      <ScrollView
        contentInsetAdjustmentBehavior="automatic"
        contentContainerClassName="pt-16 pb-36 gap-8"
      >
        {/* Header */}
        <Animated.View entering={FadeIn.delay(50).springify().damping(14)} className="px-6 gap-1">
          {isLoading ? (
            <SkeletonText lines={2} />
          ) : (
            <>
              <Text className="text-text-2 text-sm font-medium">{currentGreeting()}</Text>
              <Text className="text-text-1 font-bold text-3xl tracking-tight leading-tight">
                {user?.displayName?.split(" ")[0] ?? "Hey there"} 👋
              </Text>
            </>
          )}
        </Animated.View>

        {/* Context chips */}
        <Animated.View entering={FadeInDown.delay(110).springify().damping(16)} className="px-6">
          <View className="flex-row flex-wrap gap-2">
            <ContextChip icon="🕐" label={hourLabel} />
            <ContextChip icon="📅" label={dayLabel} />
            <ContextChip icon={LOCATION_ICONS[ctx.locationBucket]} label={ctx.locationBucket} />
            {user?.weatherEnabled && <ContextChip icon="🌤" label="Weather on" />}
          </View>
        </Animated.View>

        {/* Attune button */}
        <Animated.View entering={FadeInDown.delay(190).springify().damping(14)} className="px-6">
          <AttuneButton onPress={handleAttune} loading={isAttuning} />
        </Animated.View>

        {/* Divider */}
        <View className="px-6">
          <View className="h-px bg-surface-3" />
        </View>

        {/* Recent sessions */}
        <Animated.View
          entering={FadeInDown.delay(310).springify().damping(16)}
          className="px-6 gap-4"
        >
          <View className="flex-row items-center justify-between">
            <Text className="text-text-1 font-semibold text-lg">Recent Sessions</Text>
            {recentEvents && recentEvents.length > 0 && (
              <Text className="text-attune text-xs font-medium">
                {recentEvents.length} events
              </Text>
            )}
          </View>

          {recentEvents === undefined ? (
            <View className="gap-3">
              {[0, 1].map((i) => (
                <Skeleton key={i} style={{ height: 96, borderRadius: 16 }} />
              ))}
            </View>
          ) : sessionSummaries.length === 0 ? (
            <EmptyState />
          ) : (
            <View className="gap-3">
              {sessionSummaries.map((s, i) => (
                <SessionCard
                  key={s.ts}
                  label={formatSessionTime(s.ts)}
                  tracks={s.tracks}
                  skipPct={s.tracks > 0 ? Math.round((s.skips / s.tracks) * 100) : 0}
                  delay={i * 60}
                />
              ))}
            </View>
          )}
        </Animated.View>
      </ScrollView>

      {track && (
        <NowPlayingStrip
          trackName={track.name}
          artistName={track.artistName}
          albumArtUri={track.albumArtUri}
          isPlaying={isPlaying}
          onPlayPause={toggle}
        />
      )}
    </View>
  );
}

import { useEffect } from "react";
import { View, Text, ScrollView, Pressable } from "../../src/tw";
import { Image } from "../../src/tw/image";
import { usePlayback } from "../../contexts/PlaybackContext";
import { formatMs } from "../../lib/utils";
import Animated, {
  FadeIn,
  FadeInDown,
  useSharedValue,
  useAnimatedStyle,
  useAnimatedProps,
  withSpring,
  withRepeat,
  withTiming,
  Easing,
} from "react-native-reanimated";
import * as Haptics from "expo-haptics";
import { Platform } from "react-native";
import Svg, { Circle } from "react-native-svg";

// Animated SVG circle for the progress ring overlay on the play button
const AnimatedCircle = Animated.createAnimatedComponent(Circle);

const RING_SIZE = 78;
const RING_STROKE = 3;
const RING_RADIUS = (RING_SIZE - RING_STROKE) / 2;
const RING_CIRCUMFERENCE = 2 * Math.PI * RING_RADIUS;

function ProgressRingOverlay({ progress }: { progress: number }) {
  const offset = useSharedValue(RING_CIRCUMFERENCE);

  useEffect(() => {
    offset.value = withTiming(
      RING_CIRCUMFERENCE * (1 - Math.max(0, Math.min(1, progress))),
      { duration: 800, easing: Easing.out(Easing.cubic) }
    );
  }, [progress, offset]);

  const animatedProps = useAnimatedProps(() => ({ strokeDashoffset: offset.value }));

  return (
    <Svg
      width={RING_SIZE}
      height={RING_SIZE}
      style={{ position: "absolute" }}
    >
      <Circle
        cx={RING_SIZE / 2}
        cy={RING_SIZE / 2}
        r={RING_RADIUS}
        stroke="rgba(0,232,122,0.18)"
        strokeWidth={RING_STROKE}
        fill="none"
      />
      <AnimatedCircle
        cx={RING_SIZE / 2}
        cy={RING_SIZE / 2}
        r={RING_RADIUS}
        stroke="#00e87a"
        strokeWidth={RING_STROKE}
        fill="none"
        strokeDasharray={RING_CIRCUMFERENCE}
        animatedProps={animatedProps}
        strokeLinecap="round"
        rotation="-90"
        origin={`${RING_SIZE / 2}, ${RING_SIZE / 2}`}
      />
    </Svg>
  );
}

function ControlButton({
  icon,
  size = "md",
  primary = false,
  disabled = false,
  showProgress = false,
  progress = 0,
  onPress,
}: {
  icon: string;
  size?: "sm" | "md" | "lg";
  primary?: boolean;
  disabled?: boolean;
  showProgress?: boolean;
  progress?: number;
  onPress?: () => void;
}) {
  const scale = useSharedValue(1);
  const scaleStyle = useAnimatedStyle(() => ({ transform: [{ scale: scale.value }] }));

  const dim = size === "lg" ? 74 : size === "md" ? 54 : 42;
  const fontSize = size === "lg" ? 28 : size === "md" ? 22 : 17;

  const handlePress = async () => {
    if (disabled) return;
    if (Platform.OS === "ios") await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    scale.value = withSpring(0.88, { damping: 7, stiffness: 450 }, () => {
      scale.value = withSpring(1, { damping: 10, stiffness: 300 });
    });
    onPress?.();
  };

  return (
    <Animated.View style={[scaleStyle, { position: "relative", alignItems: "center", justifyContent: "center" }]}>
      {showProgress && <ProgressRingOverlay progress={progress} />}
      <Pressable
        onPress={handlePress}
        disabled={disabled}
        className="items-center justify-center rounded-full"
        style={{
          width: dim,
          height: dim,
          backgroundColor: primary ? "#00e87a" : "rgba(255,255,255,0.09)",
          opacity: disabled ? 0.35 : 1,
        }}
      >
        <Text style={{ fontSize, color: primary ? "#000" : "#fff", lineHeight: fontSize + 4 }}>
          {icon}
        </Text>
      </Pressable>
    </Animated.View>
  );
}

function AlbumArtPlaceholder() {
  const pulse = useSharedValue(1);

  useEffect(() => {
    pulse.value = withRepeat(
      withTiming(1.015, { duration: 2200, easing: Easing.inOut(Easing.sine) }),
      -1,
      true
    );
  }, [pulse]);

  const style = useAnimatedStyle(() => ({ transform: [{ scale: pulse.value }] }));

  return (
    <Animated.View style={[style, { width: 280, height: 280 }]}>
      <View
        className="w-full h-full rounded-3xl items-center justify-center"
        style={{
          backgroundColor: "#1a1a1a",
          borderWidth: 1,
          borderColor: "rgba(255,255,255,0.06)",
          borderCurve: "continuous",
        }}
      >
        <Text style={{ fontSize: 80, opacity: 0.25 }}>♪</Text>
      </View>
    </Animated.View>
  );
}

function ProgressBar({ progress }: { progress: number }) {
  const width = useSharedValue(0);

  useEffect(() => {
    width.value = withTiming(Math.min(Math.max(progress, 0), 1) * 100, {
      duration: 800,
      easing: Easing.out(Easing.cubic),
    });
  }, [progress, width]);

  const barStyle = useAnimatedStyle(() => ({
    width: `${width.value}%`,
  }));

  return (
    <View className="h-1.5 rounded-full bg-surface-3 overflow-hidden">
      <Animated.View
        style={[barStyle, { height: "100%", backgroundColor: "#00e87a", borderRadius: 4 }]}
      />
    </View>
  );
}

export default function PlayerScreen() {
  const { track, isPlaying, positionMs, toggle } = usePlayback();

  const progress = track ? positionMs / track.durationMs : 0;
  const noTrack = !track;

  return (
    <View className="flex-1 bg-surface-0">
      <ScrollView
        contentInsetAdjustmentBehavior="automatic"
        contentContainerClassName="px-6 pt-16 pb-32 gap-8 items-center"
      >
        {/* Album art */}
        <Animated.View entering={FadeIn.delay(50).springify().damping(14)} className="items-center">
          <View
            style={{
              boxShadow: noTrack
                ? undefined
                : "0 28px 70px rgba(0, 232, 122, 0.14), 0 0 0 1px rgba(255,255,255,0.05)",
              borderRadius: 24,
            }}
          >
            {track?.albumArtUri ? (
              <Image
                className="w-[280px] h-[280px] rounded-3xl object-cover"
                source={{ uri: track.albumArtUri }}
                transition={400}
              />
            ) : (
              <AlbumArtPlaceholder />
            )}
          </View>

          {track && (
            <Animated.View entering={FadeInDown.delay(180).springify().damping(16)}>
              <View
                className="mt-4 px-3 py-1.5 rounded-full flex-row items-center gap-1.5"
                style={{
                  backgroundColor: "rgba(0,232,122,0.1)",
                  borderWidth: 1,
                  borderColor: "rgba(0,232,122,0.22)",
                }}
              >
                <View className="w-1.5 h-1.5 rounded-full bg-attune" />
                <Text className="text-attune text-[11px] font-semibold">
                  Attuned for this moment
                </Text>
              </View>
            </Animated.View>
          )}
        </Animated.View>

        {/* Track info */}
        <Animated.View
          entering={FadeInDown.delay(120).springify().damping(16)}
          className="self-stretch gap-1"
        >
          <Text
            className="text-text-1 text-2xl font-bold tracking-tight"
            numberOfLines={1}
            selectable
          >
            {track?.name ?? "Nothing playing"}
          </Text>
          <Text className="text-text-2 text-base" numberOfLines={1}>
            {track?.artistName ?? "Connect Spotify to start"}
          </Text>
        </Animated.View>

        {/* Progress */}
        <Animated.View
          entering={FadeInDown.delay(170).springify().damping(16)}
          className="self-stretch gap-2"
        >
          <ProgressBar progress={progress} />
          <View className="flex-row justify-between">
            <Text className="text-text-3 text-xs" style={{ fontVariant: ["tabular-nums"] }}>
              {formatMs(positionMs)}
            </Text>
            <Text className="text-text-3 text-xs" style={{ fontVariant: ["tabular-nums"] }}>
              {track ? formatMs(track.durationMs) : "—:——"}
            </Text>
          </View>
        </Animated.View>

        {/* Controls */}
        <Animated.View
          entering={FadeInDown.delay(220).springify().damping(16)}
          className="self-stretch"
        >
          <View className="flex-row items-center justify-center gap-5">
            <ControlButton icon="⏮" disabled={noTrack} />
            <ControlButton
              icon={isPlaying ? "⏸" : "▶"}
              size="lg"
              primary
              disabled={noTrack}
              showProgress
              progress={progress}
              onPress={toggle}
            />
            <ControlButton icon="⏭" disabled={noTrack} />
          </View>
        </Animated.View>

        {/* Volume indicator */}
        <Animated.View
          entering={FadeInDown.delay(270).springify().damping(16)}
          className="self-stretch"
        >
          <View className="flex-row items-center gap-3">
            <Text className="text-text-3 text-xs">🔈</Text>
            <View className="flex-1 h-1 rounded-full bg-surface-3">
              <View className="w-3/4 h-full rounded-full bg-surface-4" />
            </View>
            <Text className="text-text-3 text-xs">🔊</Text>
          </View>
        </Animated.View>

        {/* Why this song */}
        <Animated.View
          entering={FadeInDown.delay(320).springify().damping(16)}
          className="self-stretch"
        >
          <View
            className="p-4 rounded-2xl gap-3"
            style={{
              backgroundColor: "rgba(0,232,122,0.06)",
              borderWidth: 1,
              borderColor: "rgba(0,232,122,0.13)",
              borderCurve: "continuous",
            }}
          >
            <View className="flex-row items-center gap-2">
              <View className="w-5 h-5 rounded-full bg-attune/20 items-center justify-center">
                <Text className="text-attune text-[10px]">✦</Text>
              </View>
              <Text className="text-attune text-xs font-bold uppercase tracking-widest">
                Why this song
              </Text>
            </View>
            <Text className="text-text-2 text-sm leading-relaxed">
              {track
                ? "High energy matches your afternoon pattern. Similar to songs you've replayed this week."
                : "Play a song to see why Attune chose it for this moment."}
            </Text>
          </View>
        </Animated.View>
      </ScrollView>
    </View>
  );
}

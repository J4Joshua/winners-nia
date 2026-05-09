import { useEffect } from "react";
import { View, Text, Pressable } from "../src/tw";
import { useRouter } from "expo-router";
import Animated, {
  FadeInDown,
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withTiming,
  withSpring,
  Easing,
} from "react-native-reanimated";
import * as Haptics from "expo-haptics";
import { Platform } from "react-native";

interface NowPlayingStripProps {
  trackName?: string;
  artistName?: string;
  albumArtUri?: string;
  isPlaying?: boolean;
  onPlayPause?: () => void;
}

function EqualizerBar({ delay = 0 }: { delay?: number }) {
  const height = useSharedValue(4);

  useEffect(() => {
    const randomDuration = 300 + Math.random() * 400;
    height.value = withRepeat(
      withTiming(4 + Math.random() * 10, {
        duration: randomDuration,
        easing: Easing.inOut(Easing.sine),
      }),
      -1,
      true
    );
  }, []);

  const style = useAnimatedStyle(() => ({ height: height.value }));

  return (
    <Animated.View
      style={[style, { width: 3, borderRadius: 2, backgroundColor: "#00e87a", alignSelf: "flex-end" }]}
    />
  );
}

export function NowPlayingStrip({
  trackName,
  artistName,
  isPlaying = false,
  onPlayPause,
}: NowPlayingStripProps) {
  const router = useRouter();
  const scale = useSharedValue(1);

  const buttonStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
  }));

  if (!trackName) return null;

  const handlePlayPause = async () => {
    if (Platform.OS === "ios") {
      await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    }
    scale.value = withSpring(0.88, { damping: 8, stiffness: 400 }, () => {
      scale.value = withSpring(1, { damping: 10, stiffness: 300 });
    });
    onPlayPause?.();
  };

  return (
    <Animated.View
      entering={FadeInDown.springify().damping(16)}
      style={{
        position: "absolute",
        bottom: 82,
        left: 12,
        right: 12,
      }}
    >
      <Pressable
        onPress={() => router.navigate("/(tabs)/player")}
        className="flex-row items-center gap-3 px-4 py-3 rounded-2xl overflow-hidden"
        style={{
          backgroundColor: "rgba(28, 28, 32, 0.95)",
          borderWidth: 1,
          borderColor: "rgba(255,255,255,0.1)",
          boxShadow: "0 8px 32px rgba(0,0,0,0.6)",
        }}
      >
        {/* Album art placeholder */}
        <View
          className="w-10 h-10 rounded-xl bg-surface-3 items-center justify-center flex-shrink-0"
          style={{ borderCurve: "continuous" }}
        >
          <Text className="text-lg">♪</Text>
        </View>

        {/* Track info */}
        <View className="flex-1 gap-0.5 overflow-hidden">
          <Text className="text-text-1 text-sm font-semibold" numberOfLines={1}>
            {trackName}
          </Text>
          {artistName && (
            <View className="flex-row items-center gap-2">
              <Text className="text-text-2 text-xs" numberOfLines={1}>
                {artistName}
              </Text>
              {isPlaying && (
                <View className="flex-row items-end gap-0.5 h-3.5">
                  <EqualizerBar />
                  <EqualizerBar delay={100} />
                  <EqualizerBar delay={200} />
                </View>
              )}
            </View>
          )}
        </View>

        {/* Play/pause */}
        <Animated.View style={buttonStyle}>
          <Pressable
            onPress={handlePlayPause}
            className="w-9 h-9 rounded-full bg-attune items-center justify-center"
          >
            <Text className="text-surface-0 text-sm font-bold">
              {isPlaying ? "⏸" : "▶"}
            </Text>
          </Pressable>
        </Animated.View>
      </Pressable>
    </Animated.View>
  );
}

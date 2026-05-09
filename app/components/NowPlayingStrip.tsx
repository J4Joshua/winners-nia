import { useCallback, useEffect, useRef } from "react";
import { View, Text, Pressable } from "../src/tw";
import { Image } from "../src/tw/image";
import { useRouter } from "expo-router";
import Animated, {
  FadeInDown,
  FadeOutDown,
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  withRepeat,
  withTiming,
  Easing,
} from "react-native-reanimated";
import * as Haptics from "expo-haptics";
import { Platform } from "react-native";

interface EqualizerBarProps {
  baseHeight?: number;
  delay?: number;
}

function EqualizerBar({ baseHeight = 4, delay = 0 }: EqualizerBarProps) {
  const height = useSharedValue(baseHeight);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const duration = 280 + Math.random() * 300;
    const target = baseHeight + 3 + Math.random() * 9;

    timerRef.current = setTimeout(() => {
      timerRef.current = null;
      height.value = withRepeat(
        withTiming(target, { duration, easing: Easing.inOut(Easing.sine) }),
        -1,
        true
      );
    }, delay);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
    // Intentionally omit Math.random() results from deps — they should only compute once
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baseHeight, delay]);

  const style = useAnimatedStyle(() => ({
    height: height.value,
    width: 3,
    borderRadius: 2,
    backgroundColor: "#00e87a",
    alignSelf: "flex-end",
  }));

  return <Animated.View style={style} />;
}

interface NowPlayingStripProps {
  trackName: string;
  artistName?: string;
  albumArtUri?: string;
  isPlaying?: boolean;
  onPlayPause?: () => void;
}

export function NowPlayingStrip({
  trackName,
  artistName,
  albumArtUri,
  isPlaying = false,
  onPlayPause,
}: NowPlayingStripProps) {
  const router = useRouter();
  const btnScale = useSharedValue(1);
  const btnStyle = useAnimatedStyle(() => ({ transform: [{ scale: btnScale.value }] }));

  const handlePlayPause = useCallback(async () => {
    if (Platform.OS === "ios") await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    btnScale.value = withSpring(0.86, { damping: 7, stiffness: 450 }, () => {
      btnScale.value = withSpring(1, { damping: 10, stiffness: 300 });
    });
    onPlayPause?.();
  }, [onPlayPause, btnScale]);

  const handleNavigate = useCallback(() => {
    router.navigate("/(tabs)/player");
  }, [router]);

  return (
    <Animated.View
      entering={FadeInDown.springify().damping(16)}
      exiting={FadeOutDown.springify().damping(16)}
      style={{ position: "absolute", bottom: 86, left: 10, right: 10 }}
    >
      <Pressable
        onPress={handleNavigate}
        style={{
          flexDirection: "row",
          alignItems: "center",
          gap: 12,
          paddingHorizontal: 14,
          paddingVertical: 10,
          borderRadius: 20,
          backgroundColor: "rgba(22, 22, 26, 0.97)",
          borderWidth: 1,
          borderColor: "rgba(255,255,255,0.1)",
          boxShadow: "0 10px 40px rgba(0,0,0,0.7), 0 0 0 1px rgba(255,255,255,0.05)",
        }}
      >
        {/* Album art */}
        <View
          style={{
            width: 42,
            height: 42,
            borderRadius: 10,
            backgroundColor: "#1a1a1a",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
            overflow: "hidden",
            borderCurve: "continuous",
          }}
        >
          {albumArtUri ? (
            <Image
              source={{ uri: albumArtUri }}
              style={{ width: 42, height: 42 }}
              contentFit="cover"
            />
          ) : (
            <Text style={{ fontSize: 18 }}>♪</Text>
          )}
        </View>

        {/* Track info */}
        <View style={{ flex: 1, gap: 2, overflow: "hidden" }}>
          <Text className="text-text-1 text-[13px] font-semibold" numberOfLines={1}>
            {trackName}
          </Text>
          {artistName && (
            <View style={{ flexDirection: "row", alignItems: "center", gap: 6 }}>
              <Text className="text-text-2 text-xs" numberOfLines={1} style={{ flex: 1 }}>
                {artistName}
              </Text>
              {isPlaying && (
                <View style={{ flexDirection: "row", alignItems: "flex-end", gap: 2, height: 14 }}>
                  <EqualizerBar delay={0} />
                  <EqualizerBar delay={120} />
                  <EqualizerBar delay={240} />
                  <EqualizerBar delay={60} />
                </View>
              )}
            </View>
          )}
        </View>

        {/* Play/pause */}
        <Animated.View style={btnStyle}>
          <Pressable
            onPress={handlePlayPause}
            style={{
              width: 38,
              height: 38,
              borderRadius: 19,
              backgroundColor: "#00e87a",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
            }}
          >
            <Text style={{ fontSize: 16, color: "#000", lineHeight: 20 }}>
              {isPlaying ? "⏸" : "▶"}
            </Text>
          </Pressable>
        </Animated.View>
      </Pressable>
    </Animated.View>
  );
}

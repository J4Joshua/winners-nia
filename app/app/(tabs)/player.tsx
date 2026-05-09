import { View, Text, ScrollView, Pressable } from "../../src/tw";
import { Image } from "../../src/tw/image";
import { ProgressRing } from "../../components/ProgressRing";
import Animated, {
  FadeIn,
  FadeInDown,
  useSharedValue,
  useAnimatedStyle,
  withSpring,
} from "react-native-reanimated";
import * as Haptics from "expo-haptics";
import { Platform } from "react-native";

function ControlButton({
  icon,
  size = "md",
  primary = false,
  onPress,
}: {
  icon: string;
  size?: "sm" | "md" | "lg";
  primary?: boolean;
  onPress?: () => void;
}) {
  const scale = useSharedValue(1);
  const style = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
  }));

  const dim = size === "lg" ? 72 : size === "md" ? 52 : 40;
  const fontSize = size === "lg" ? 28 : size === "md" ? 22 : 16;

  const handlePress = async () => {
    if (Platform.OS === "ios") {
      await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    }
    scale.value = withSpring(0.92, { damping: 8, stiffness: 400 }, () => {
      scale.value = withSpring(1, { damping: 10, stiffness: 300 });
    });
    onPress?.();
  };

  return (
    <Animated.View style={style}>
      <Pressable
        onPress={handlePress}
        className="items-center justify-center rounded-full"
        style={{
          width: dim,
          height: dim,
          backgroundColor: primary ? "#00e87a" : "rgba(255,255,255,0.08)",
        }}
      >
        <Text
          style={{
            fontSize,
            color: primary ? "#000" : "#fff",
          }}
        >
          {icon}
        </Text>
      </Pressable>
    </Animated.View>
  );
}

export default function PlayerScreen() {
  const albumArt = null as string | null;

  return (
    <View className="flex-1 bg-surface-0">
      <ScrollView
        contentInsetAdjustmentBehavior="automatic"
        contentContainerClassName="px-6 pt-16 pb-32 gap-8"
      >
        {/* Album art */}
        <Animated.View
          entering={FadeIn.delay(50).springify().damping(14)}
          className="items-center"
        >
          <View
            className="w-72 h-72 rounded-3xl items-center justify-center"
            style={{
              backgroundColor: "#1a1a1a",
              boxShadow: "0 24px 60px rgba(0, 232, 122, 0.12)",
              borderWidth: 1,
              borderColor: "rgba(255,255,255,0.06)",
              borderCurve: "continuous",
            }}
          >
            {albumArt ? (
              <Image
                className="w-full h-full rounded-3xl object-cover"
                source={{ uri: albumArt }}
              />
            ) : (
              <Text className="text-7xl">♪</Text>
            )}
          </View>

          {/* Attune badge */}
          <View
            className="mt-4 px-3 py-1.5 rounded-full flex-row items-center gap-1.5"
            style={{ backgroundColor: "rgba(0,232,122,0.12)", borderWidth: 1, borderColor: "rgba(0,232,122,0.25)" }}
          >
            <View className="w-1.5 h-1.5 rounded-full bg-attune" />
            <Text className="text-attune text-xs font-medium">Attuned for this moment</Text>
          </View>
        </Animated.View>

        {/* Track info */}
        <Animated.View
          entering={FadeInDown.delay(120).springify().damping(16)}
          className="gap-1"
        >
          <Text className="text-text-1 text-2xl font-bold tracking-tight" selectable>
            Nothing playing
          </Text>
          <Text className="text-text-2 text-base">Connect Spotify to start</Text>
        </Animated.View>

        {/* Progress */}
        <Animated.View entering={FadeInDown.delay(180).springify().damping(16)}>
          <View className="gap-2">
            <View className="h-1 rounded-full bg-surface-3 overflow-hidden">
              <View className="h-full w-1/3 rounded-full bg-attune" />
            </View>
            <View className="flex-row justify-between">
              <Text className="text-text-3 text-xs" style={{ fontVariant: ["tabular-nums"] }}>0:00</Text>
              <Text className="text-text-3 text-xs" style={{ fontVariant: ["tabular-nums"] }}>—:——</Text>
            </View>
          </View>
        </Animated.View>

        {/* Controls */}
        <Animated.View entering={FadeInDown.delay(240).springify().damping(16)}>
          <View className="flex-row items-center justify-center gap-6">
            <ControlButton icon="⏮" />
            <ControlButton icon="▶" size="lg" primary />
            <ControlButton icon="⏭" />
          </View>
        </Animated.View>

        {/* Volume / extra */}
        <Animated.View entering={FadeInDown.delay(300).springify().damping(16)}>
          <View className="flex-row items-center gap-3">
            <Text className="text-text-3 text-xs">🔈</Text>
            <View className="flex-1 h-1 rounded-full bg-surface-3" />
            <Text className="text-text-3 text-xs">🔊</Text>
          </View>
        </Animated.View>

        {/* Explainability */}
        <Animated.View entering={FadeInDown.delay(360).springify().damping(16)}>
          <View
            className="p-4 rounded-2xl gap-2"
            style={{ backgroundColor: "rgba(0,232,122,0.06)", borderWidth: 1, borderColor: "rgba(0,232,122,0.12)" }}
          >
            <Text className="text-attune text-xs font-semibold uppercase tracking-widest">Why this song</Text>
            <Text className="text-text-2 text-sm leading-relaxed">
              Your energy tends to peak in the afternoon. High-tempo tracks from your recent favorites.
            </Text>
          </View>
        </Animated.View>
      </ScrollView>
    </View>
  );
}

import { useEffect, useRef } from "react";
import { useRouter } from "expo-router";
import { View, Text, Pressable } from "../src/tw";
import { useSpotifyAuth } from "../hooks/useSpotifyAuth";
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  withRepeat,
  withSequence,
  withTiming,
  FadeIn,
  FadeInDown,
} from "react-native-reanimated";
import * as Haptics from "expo-haptics";
import { Platform } from "react-native";
import { Image } from "../src/tw/image";

const AnimatedView = Animated.createAnimatedComponent(View);

function SpotifyIcon() {
  return (
    <Image
      className="w-5 h-5"
      source="sf:music.note"
      tintColor="#000"
    />
  );
}

function PulseRing({ delay = 0 }: { delay?: number }) {
  const scale = useSharedValue(0.8);
  const opacity = useSharedValue(0.6);

  useEffect(() => {
    scale.value = withRepeat(
      withSequence(
        withTiming(1, { duration: 0 }),
        withSpring(1.8, { damping: 6, stiffness: 40 })
      ),
      -1,
      false
    );
    opacity.value = withRepeat(
      withSequence(
        withTiming(0.5, { duration: 0 }),
        withTiming(0, { duration: 1800 })
      ),
      -1,
      false
    );
  }, []);

  const style = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
    opacity: opacity.value,
  }));

  return (
    <Animated.View
      style={[
        style,
        {
          position: "absolute",
          width: 88,
          height: 88,
          borderRadius: 44,
          backgroundColor: "#00e87a",
        },
      ]}
    />
  );
}

export default function LoginScreen() {
  const router = useRouter();
  const { login, loading, error, userId, loadStoredUserId } = useSpotifyAuth();

  const buttonScale = useSharedValue(1);
  const buttonStyle = useAnimatedStyle(() => ({
    transform: [{ scale: buttonScale.value }],
  }));

  useEffect(() => {
    loadStoredUserId().then((id) => {
      if (id) router.replace("/(tabs)");
    });
  }, []);

  useEffect(() => {
    if (userId) router.replace("/(tabs)");
  }, [userId]);

  const handleLogin = async () => {
    if (Platform.OS === "ios") {
      await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    }
    buttonScale.value = withSpring(0.96, { damping: 10, stiffness: 300 }, () => {
      buttonScale.value = withSpring(1, { damping: 12, stiffness: 300 });
    });
    await login();
  };

  return (
    <View className="flex-1 bg-surface-0 items-center justify-center px-8">
      {/* Ambient glow */}
      <View
        style={{
          position: "absolute",
          top: "25%",
          width: 280,
          height: 280,
          borderRadius: 140,
          backgroundColor: "rgba(0, 232, 122, 0.07)",
        }}
      />

      {/* Logo mark */}
      <AnimatedView
        entering={FadeIn.delay(100).springify().damping(14)}
        className="items-center mb-16"
      >
        <View className="relative items-center justify-center w-[88px] h-[88px] mb-8">
          <PulseRing />
          <View className="w-[72px] h-[72px] rounded-full bg-attune items-center justify-center">
            <Text className="text-3xl" style={{ fontFamily: "ui-rounded" }}>♪</Text>
          </View>
        </View>

        <Text className="text-text-1 font-bold text-4xl tracking-tight mb-2">
          Attune
        </Text>
        <Text className="text-text-2 text-base text-center leading-relaxed">
          Music that fits{"\n"}this exact moment
        </Text>
      </AnimatedView>

      {/* Features */}
      <AnimatedView
        entering={FadeInDown.delay(250).springify().damping(16)}
        className="gap-3 mb-16 w-full"
      >
        {[
          { icon: "🎯", label: "Learns from your skips and replays" },
          { icon: "📍", label: "Adapts to your time and place" },
          { icon: "⚡", label: "Gets better with every session" },
        ].map(({ icon, label }) => (
          <View key={label} className="flex-row items-center gap-3">
            <View className="w-8 h-8 rounded-full bg-surface-3 items-center justify-center">
              <Text className="text-sm">{icon}</Text>
            </View>
            <Text className="text-text-2 text-sm flex-1">{label}</Text>
          </View>
        ))}
      </AnimatedView>

      {/* CTA */}
      <AnimatedView
        entering={FadeInDown.delay(380).springify().damping(16)}
        className="w-full gap-3"
      >
        <Animated.View style={buttonStyle}>
          <Pressable
            onPress={handleLogin}
            disabled={loading}
            className="w-full h-[54px] bg-attune rounded-2xl flex-row items-center justify-center gap-2"
            style={{ opacity: loading ? 0.7 : 1 }}
          >
            <Text className="text-surface-0 font-semibold text-base tracking-tight">
              {loading ? "Connecting…" : "Continue with Spotify"}
            </Text>
          </Pressable>
        </Animated.View>

        {error && (
          <Text className="text-sf-red text-sm text-center">{error}</Text>
        )}

        <Text className="text-text-3 text-xs text-center px-4">
          We never store your listening history raw. Data stays on-device and in your private Convex deployment.
        </Text>
      </AnimatedView>
    </View>
  );
}

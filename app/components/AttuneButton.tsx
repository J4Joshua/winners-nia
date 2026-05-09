import { useEffect, useRef } from "react";
import { View, Text, Pressable } from "../src/tw";
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  withRepeat,
  withSequence,
  withTiming,
  Easing,
} from "react-native-reanimated";
import * as Haptics from "expo-haptics";
import { Platform } from "react-native";

interface AttuneButtonProps {
  onPress: () => void | Promise<void>;
  loading?: boolean;
}

function PulseRings() {
  const ring1 = useSharedValue(1);
  const ring2 = useSharedValue(1);
  const op1 = useSharedValue(0.35);
  const op2 = useSharedValue(0.35);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    ring1.value = withRepeat(
      withSequence(
        withTiming(1, { duration: 0 }),
        withTiming(1.6, { duration: 1400, easing: Easing.out(Easing.quad) })
      ),
      -1,
      false
    );
    op1.value = withRepeat(
      withSequence(withTiming(0.35, { duration: 0 }), withTiming(0, { duration: 1400 })),
      -1,
      false
    );

    timerRef.current = setTimeout(() => {
      ring2.value = withRepeat(
        withSequence(
          withTiming(1, { duration: 0 }),
          withTiming(1.6, { duration: 1400, easing: Easing.out(Easing.quad) })
        ),
        -1,
        false
      );
      op2.value = withRepeat(
        withSequence(withTiming(0.35, { duration: 0 }), withTiming(0, { duration: 1400 })),
        -1,
        false
      );
    }, 700);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  const s1 = useAnimatedStyle(() => ({
    transform: [{ scale: ring1.value }],
    opacity: op1.value,
  }));
  const s2 = useAnimatedStyle(() => ({
    transform: [{ scale: ring2.value }],
    opacity: op2.value,
  }));

  return (
    <>
      <Animated.View
        style={[s1, { position: "absolute", width: 160, height: 160, borderRadius: 80, backgroundColor: "#00e87a" }]}
      />
      <Animated.View
        style={[s2, { position: "absolute", width: 160, height: 160, borderRadius: 80, backgroundColor: "#00e87a" }]}
      />
    </>
  );
}

function LoadingSpinner() {
  const rotation = useSharedValue(0);

  useEffect(() => {
    rotation.value = withRepeat(
      withTiming(360, { duration: 1000, easing: Easing.linear }),
      -1,
      false
    );
  }, []);

  const style = useAnimatedStyle(() => ({
    transform: [{ rotate: `${rotation.value}deg` }],
  }));

  return (
    <Animated.View
      style={[
        style,
        {
          width: 24,
          height: 24,
          borderRadius: 12,
          borderWidth: 2.5,
          borderColor: "transparent",
          borderTopColor: "#000",
        },
      ]}
    />
  );
}

export function AttuneButton({ onPress, loading = false }: AttuneButtonProps) {
  const scale = useSharedValue(1);
  const buttonStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
  }));

  const handlePress = async () => {
    if (loading) return;
    if (Platform.OS === "ios") {
      await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
    }
    scale.value = withSpring(0.93, { damping: 8, stiffness: 350 }, () => {
      scale.value = withSpring(1, { damping: 10, stiffness: 280 });
    });
    await onPress();
  };

  return (
    <View className="items-center gap-6">
      {/* Button with glow rings */}
      <View className="items-center justify-center" style={{ height: 180 }}>
        {!loading && <PulseRings />}

        <Animated.View style={buttonStyle}>
          <Pressable
            onPress={handlePress}
            disabled={loading}
            style={{
              width: 148,
              height: 148,
              borderRadius: 74,
              backgroundColor: "#00e87a",
              alignItems: "center",
              justifyContent: "center",
              boxShadow: loading ? undefined : "0 0 40px rgba(0, 232, 122, 0.4)",
            }}
          >
            {loading ? (
              <LoadingSpinner />
            ) : (
              <View className="items-center gap-1">
                <Text className="text-5xl">♪</Text>
              </View>
            )}
          </Pressable>
        </Animated.View>
      </View>

      {/* Label */}
      <View className="items-center gap-1">
        <Text className="text-text-1 font-bold text-xl tracking-tight">
          {loading ? "Attuning…" : "Attune this moment"}
        </Text>
        <Text className="text-text-2 text-sm text-center">
          {loading
            ? "Finding songs that fit right now"
            : "Tap to generate your perfect playlist"}
        </Text>
      </View>
    </View>
  );
}

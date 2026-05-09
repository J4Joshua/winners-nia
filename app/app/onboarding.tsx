import { useEffect } from "react";
import { useLocalSearchParams, useRouter } from "expo-router";
import { useQuery } from "convex/react";
import { api } from "../convex/_generated/api";
import type { Id } from "../convex/_generated/dataModel";
import { View, Text } from "../src/tw";
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withTiming,
  withSpring,
  FadeIn,
  FadeInDown,
  Easing,
} from "react-native-reanimated";

const STEPS = [
  { key: "fetching", label: "Fetching your Spotify history", icon: "📡" },
  { key: "analyzing", label: "Analyzing your taste", icon: "🧠" },
  { key: "training", label: "Training your personal model", icon: "⚡" },
  { key: "ready", label: "You're ready!", icon: "🎉" },
];

function RotatingRing() {
  const rotation = useSharedValue(0);

  useEffect(() => {
    rotation.value = withRepeat(
      withTiming(360, { duration: 2000, easing: Easing.linear }),
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
          width: 80,
          height: 80,
          borderRadius: 40,
          borderWidth: 2,
          borderColor: "transparent",
          borderTopColor: "#00e87a",
          borderRightColor: "rgba(0,232,122,0.3)",
        },
      ]}
    />
  );
}

function StepRow({
  step,
  currentStatus,
  index,
}: {
  step: (typeof STEPS)[number];
  currentStatus: string;
  index: number;
}) {
  const stepIndex = STEPS.findIndex((s) => s.key === currentStatus);
  const isDone = index < stepIndex;
  const isActive = index === stepIndex;
  const isPending = index > stepIndex;

  const scale = useSharedValue(1);
  useEffect(() => {
    if (isActive) {
      scale.value = withRepeat(
        withTiming(1.04, { duration: 800, easing: Easing.inOut(Easing.sine) }),
        -1,
        true
      );
    } else {
      scale.value = withSpring(1);
    }
  }, [isActive]);

  const style = useAnimatedStyle(() => ({ transform: [{ scale: scale.value }] }));

  return (
    <Animated.View
      entering={FadeInDown.delay(index * 80).springify().damping(16)}
      style={style}
    >
      <View
        className="flex-row items-center gap-4 p-4 rounded-2xl"
        style={{
          backgroundColor: isActive
            ? "rgba(0, 232, 122, 0.1)"
            : isDone
            ? "rgba(0, 232, 122, 0.04)"
            : "rgba(255,255,255,0.04)",
          borderWidth: 1,
          borderColor: isActive
            ? "rgba(0, 232, 122, 0.3)"
            : isDone
            ? "rgba(0,232,122,0.12)"
            : "rgba(255,255,255,0.06)",
        }}
      >
        <View className="w-10 h-10 rounded-full items-center justify-center bg-surface-2">
          <Text className="text-lg">{isDone ? "✓" : step.icon}</Text>
        </View>
        <View className="flex-1 gap-0.5">
          <Text
            className="text-sm font-medium"
            style={{
              color: isActive ? "#00e87a" : isDone ? "rgba(255,255,255,0.7)" : "rgba(255,255,255,0.35)",
            }}
          >
            {step.label}
          </Text>
        </View>
        {isActive && (
          <View className="w-2 h-2 rounded-full bg-attune" />
        )}
        {isDone && (
          <Text className="text-attune text-xs font-medium">Done</Text>
        )}
      </View>
    </Animated.View>
  );
}

export default function OnboardingScreen() {
  const router = useRouter();
  const { userId } = useLocalSearchParams<{ userId: string }>();

  const job = useQuery(
    api.queries.getOnboardingJob,
    userId ? { userId: userId as Id<"users"> } : "skip"
  );

  const status = job?.status ?? "fetching";

  useEffect(() => {
    if (status === "ready") {
      const t = setTimeout(() => router.replace("/(tabs)"), 1500);
      return () => clearTimeout(t);
    }
  }, [status]);

  return (
    <View className="flex-1 bg-surface-0 px-6 pt-20">
      <Animated.View
        entering={FadeIn.delay(50).springify().damping(14)}
        className="items-center mb-12"
      >
        <View className="relative items-center justify-center w-20 h-20 mb-6">
          <RotatingRing />
          <View className="absolute w-14 h-14 rounded-full bg-surface-2 items-center justify-center">
            <Text className="text-2xl">
              {status === "ready" ? "🎉" : "♪"}
            </Text>
          </View>
        </View>

        <Text className="text-text-1 text-2xl font-bold text-center mb-1">
          {status === "ready" ? "You're all set" : "Setting up Attune"}
        </Text>
        <Text className="text-text-2 text-sm text-center">
          {status === "ready"
            ? "Your first playlist is ready"
            : "This usually takes under a minute"}
        </Text>
      </Animated.View>

      <View className="gap-3">
        {STEPS.map((step, i) => (
          <StepRow key={step.key} step={step} currentStatus={status} index={i} />
        ))}
      </View>

      {job?.errorMessage && (
        <View className="mt-6 p-4 rounded-2xl bg-surface-2 border border-sf-red/20">
          <Text className="text-sf-red text-sm">{job.errorMessage}</Text>
        </View>
      )}
    </View>
  );
}

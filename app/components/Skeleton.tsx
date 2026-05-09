import { useEffect } from "react";
import { View } from "../src/tw";
import type { ViewProps } from "../src/tw";
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withTiming,
  Easing,
} from "react-native-reanimated";

interface SkeletonProps {
  className?: string;
  style?: ViewProps["style"];
}

export function Skeleton({ className, style }: SkeletonProps) {
  const opacity = useSharedValue(0.4);

  useEffect(() => {
    opacity.value = withRepeat(
      withTiming(0.12, { duration: 900, easing: Easing.inOut(Easing.sine) }),
      -1,
      true
    );
  }, []);

  const animStyle = useAnimatedStyle(() => ({ opacity: opacity.value }));

  return (
    <Animated.View
      style={[animStyle, { backgroundColor: "#fff", borderRadius: 10 }, style]}
      className={className}
    />
  );
}

export function SkeletonText({ lines = 1, className }: { lines?: number; className?: string }) {
  return (
    <View className={`gap-2 ${className ?? ""}`}>
      {Array.from({ length: lines }, (_, i) => (
        <Skeleton
          key={i}
          style={{
            height: 14,
            width: i === lines - 1 && lines > 1 ? "65%" : "100%",
            borderRadius: 7,
          }}
        />
      ))}
    </View>
  );
}

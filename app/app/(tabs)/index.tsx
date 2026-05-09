import { useEffect, useState } from "react";
import { ScrollView, View, Text } from "../../src/tw";
import { AttuneButton } from "../../components/AttuneButton";
import { NowPlayingStrip } from "../../components/NowPlayingStrip";
import { useRouter } from "expo-router";
import Animated, {
  FadeIn,
  FadeInDown,
} from "react-native-reanimated";
import { buildContextSnapshot } from "../../lib/context";

function GreetingText() {
  const hour = new Date().getHours();
  const greeting =
    hour < 5 ? "Late night" :
    hour < 12 ? "Good morning" :
    hour < 17 ? "Good afternoon" :
    hour < 21 ? "Good evening" : "Good night";

  return <>{greeting}</>;
}

function ContextPill({ icon, label }: { icon: string; label: string }) {
  return (
    <View className="flex-row items-center gap-1.5 px-3 py-1.5 rounded-full bg-surface-3">
      <Text className="text-xs">{icon}</Text>
      <Text className="text-text-2 text-xs">{label}</Text>
    </View>
  );
}

export default function HomeScreen() {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);
  const ctx = buildContextSnapshot();

  const hourLabel = `${ctx.hourLocal}:00`;
  const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  const dayLabel = days[ctx.dow];

  const handleAttune = async () => {
    setIsLoading(true);
    // TODO: call generatePlaylist Convex action
    await new Promise((r) => setTimeout(r, 1800));
    setIsLoading(false);
  };

  return (
    <View className="flex-1 bg-surface-0">
      <ScrollView
        contentInsetAdjustmentBehavior="automatic"
        contentContainerClassName="px-6 pt-16 pb-32 gap-8"
      >
        {/* Header */}
        <Animated.View entering={FadeIn.delay(50).springify().damping(14)}>
          <View className="gap-1">
            <Text className="text-text-2 text-sm font-medium">
              <GreetingText />
            </Text>
            <Text className="text-text-1 font-bold text-3xl tracking-tight">
              What&apos;s your moment?
            </Text>
          </View>
        </Animated.View>

        {/* Context chips */}
        <Animated.View entering={FadeInDown.delay(120).springify().damping(16)}>
          <View className="flex-row flex-wrap gap-2">
            <ContextPill icon="🕐" label={hourLabel} />
            <ContextPill icon="📅" label={dayLabel} />
            <ContextPill icon="📍" label={ctx.locationBucket} />
          </View>
        </Animated.View>

        {/* Main attune button */}
        <Animated.View entering={FadeInDown.delay(200).springify().damping(14)}>
          <AttuneButton onPress={handleAttune} loading={isLoading} />
        </Animated.View>

        {/* Divider */}
        <Animated.View entering={FadeInDown.delay(280).springify().damping(16)}>
          <View className="h-px bg-surface-3" />
        </Animated.View>

        {/* Recent sessions placeholder */}
        <Animated.View entering={FadeInDown.delay(340).springify().damping(16)}>
          <View className="gap-4">
            <Text className="text-text-1 font-semibold text-lg">Recent Sessions</Text>
            <View className="gap-3">
              {[
                { time: "This morning", tracks: 14, skipRate: "12% skipped" },
                { time: "Yesterday, 10pm", tracks: 22, skipRate: "8% skipped" },
              ].map((session) => (
                <View
                  key={session.time}
                  className="p-4 rounded-2xl bg-surface-2 gap-1"
                  style={{ borderWidth: 1, borderColor: "rgba(255,255,255,0.06)" }}
                >
                  <View className="flex-row items-center justify-between">
                    <Text className="text-text-1 text-sm font-medium">{session.time}</Text>
                    <Text className="text-attune text-xs font-medium">{session.skipRate}</Text>
                  </View>
                  <Text className="text-text-3 text-xs">{session.tracks} tracks</Text>
                </View>
              ))}
            </View>
          </View>
        </Animated.View>
      </ScrollView>

      {/* Floating now playing strip */}
      <NowPlayingStrip />
    </View>
  );
}

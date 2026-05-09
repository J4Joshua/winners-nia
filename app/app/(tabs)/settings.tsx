import { ScrollView, View, Text, Pressable } from "../../src/tw";
import Animated, { FadeIn, FadeInDown } from "react-native-reanimated";
import { useRouter } from "expo-router";
import * as Haptics from "expo-haptics";
import { Platform, Switch } from "react-native";
import { useState } from "react";

type LocationBucket = "home" | "gym" | "transit" | "other";

const LOCATION_OPTIONS: { value: LocationBucket; label: string; icon: string }[] = [
  { value: "home", label: "Home", icon: "🏠" },
  { value: "gym", label: "Gym", icon: "💪" },
  { value: "transit", label: "Transit", icon: "🚇" },
  { value: "other", label: "Other", icon: "📍" },
];

function SectionHeader({ title }: { title: string }) {
  return (
    <Text className="text-text-3 text-xs font-semibold uppercase tracking-widest px-1">
      {title}
    </Text>
  );
}

function SettingsRow({
  icon,
  label,
  sublabel,
  right,
  onPress,
  danger = false,
}: {
  icon: string;
  label: string;
  sublabel?: string;
  right?: React.ReactNode;
  onPress?: () => void;
  danger?: boolean;
}) {
  return (
    <Pressable
      onPress={onPress}
      className="flex-row items-center gap-3 px-4 py-3.5"
      style={{ opacity: onPress ? 1 : 0.6 }}
    >
      <View className="w-9 h-9 rounded-xl bg-surface-3 items-center justify-center">
        <Text className="text-base">{icon}</Text>
      </View>
      <View className="flex-1 gap-0.5">
        <Text
          className="text-sm font-medium"
          style={{ color: danger ? "#ff453a" : "#ffffff" }}
        >
          {label}
        </Text>
        {sublabel && (
          <Text className="text-text-3 text-xs">{sublabel}</Text>
        )}
      </View>
      {right}
    </Pressable>
  );
}

function Divider() {
  return <View className="h-px bg-surface-3 ml-16" />;
}

export default function SettingsScreen() {
  const router = useRouter();
  const [locationOverride, setLocationOverride] = useState<LocationBucket | null>(null);
  const [weatherEnabled, setWeatherEnabled] = useState(false);

  const handleLocationSelect = async (value: LocationBucket) => {
    if (Platform.OS === "ios") {
      await Haptics.selectionAsync();
    }
    setLocationOverride(value === locationOverride ? null : value);
  };

  const handleLogout = async () => {
    if (Platform.OS === "ios") {
      await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
    }
    // TODO: clear stored userId + call logout
    router.replace("/login");
  };

  return (
    <View className="flex-1 bg-surface-0">
      <ScrollView
        contentInsetAdjustmentBehavior="automatic"
        contentContainerClassName="px-6 pt-16 pb-32 gap-6"
      >
        <Animated.View entering={FadeIn.delay(50).springify().damping(14)}>
          <Text className="text-text-1 font-bold text-3xl tracking-tight">Settings</Text>
        </Animated.View>

        {/* Account */}
        <Animated.View entering={FadeInDown.delay(100).springify().damping(16)} className="gap-3">
          <SectionHeader title="Account" />
          <View className="rounded-2xl bg-surface-2 overflow-hidden" style={{ borderWidth: 1, borderColor: "rgba(255,255,255,0.06)" }}>
            <SettingsRow
              icon="🎵"
              label="Spotify Account"
              sublabel="Connected"
              right={<View className="w-2 h-2 rounded-full bg-attune" />}
            />
          </View>
        </Animated.View>

        {/* Location */}
        <Animated.View entering={FadeInDown.delay(160).springify().damping(16)} className="gap-3">
          <SectionHeader title="Location" />
          <View className="rounded-2xl bg-surface-2 overflow-hidden" style={{ borderWidth: 1, borderColor: "rgba(255,255,255,0.06)" }}>
            <View className="px-4 py-3 gap-3">
              <View className="gap-1">
                <Text className="text-text-1 text-sm font-medium">Override location</Text>
                <Text className="text-text-3 text-xs">
                  Let Attune know where you are for better picks
                </Text>
              </View>
              <View className="flex-row flex-wrap gap-2">
                {LOCATION_OPTIONS.map(({ value, label, icon }) => (
                  <Pressable
                    key={value}
                    onPress={() => handleLocationSelect(value)}
                    className="flex-row items-center gap-1.5 px-3 py-2 rounded-xl"
                    style={{
                      backgroundColor:
                        locationOverride === value
                          ? "rgba(0,232,122,0.15)"
                          : "rgba(255,255,255,0.06)",
                      borderWidth: 1,
                      borderColor:
                        locationOverride === value
                          ? "rgba(0,232,122,0.4)"
                          : "rgba(255,255,255,0.08)",
                    }}
                  >
                    <Text className="text-sm">{icon}</Text>
                    <Text
                      className="text-xs font-medium"
                      style={{ color: locationOverride === value ? "#00e87a" : "rgba(255,255,255,0.7)" }}
                    >
                      {label}
                    </Text>
                  </Pressable>
                ))}
              </View>
            </View>
          </View>
        </Animated.View>

        {/* Personalization */}
        <Animated.View entering={FadeInDown.delay(220).springify().damping(16)} className="gap-3">
          <SectionHeader title="Personalization" />
          <View className="rounded-2xl bg-surface-2 overflow-hidden" style={{ borderWidth: 1, borderColor: "rgba(255,255,255,0.06)" }}>
            <SettingsRow
              icon="🌤"
              label="Weather-aware picks"
              sublabel="Uses your local weather to tune energy"
              right={
                <Switch
                  value={weatherEnabled}
                  onValueChange={setWeatherEnabled}
                  trackColor={{ false: "#333", true: "#00e87a" }}
                  thumbColor="#fff"
                />
              }
            />
          </View>
        </Animated.View>

        {/* Privacy */}
        <Animated.View entering={FadeInDown.delay(280).springify().damping(16)} className="gap-3">
          <SectionHeader title="Privacy" />
          <View
            className="p-4 rounded-2xl gap-2 bg-surface-2"
            style={{ borderWidth: 1, borderColor: "rgba(255,255,255,0.06)" }}
          >
            <Text className="text-text-1 text-sm font-medium">Your data stays yours</Text>
            <Text className="text-text-3 text-xs leading-relaxed">
              Attune stores your listening model in your private Convex deployment.
              No raw GPS coordinates are ever saved — only coarse location buckets (home, gym, transit).
              You can delete all data at any time.
            </Text>
          </View>
        </Animated.View>

        {/* Danger */}
        <Animated.View entering={FadeInDown.delay(340).springify().damping(16)} className="gap-3">
          <SectionHeader title="Account" />
          <View className="rounded-2xl bg-surface-2 overflow-hidden" style={{ borderWidth: 1, borderColor: "rgba(255,255,255,0.06)" }}>
            <SettingsRow
              icon="🚪"
              label="Disconnect Spotify"
              danger
              onPress={handleLogout}
            />
          </View>
        </Animated.View>

        <Animated.View entering={FadeInDown.delay(400).springify().damping(16)}>
          <Text className="text-text-3 text-xs text-center">Attune v1.0.0 · Built with ♥</Text>
        </Animated.View>
      </ScrollView>
    </View>
  );
}

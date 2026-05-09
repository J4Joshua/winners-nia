import { useCallback, useState } from "react";
import { ScrollView, View, Text, Pressable } from "../../src/tw";
import { Image } from "../../src/tw/image";
import { Skeleton } from "../../components/Skeleton";
import { useAuth } from "../../contexts/AuthContext";
import { useQuery } from "convex/react";
import { api } from "../../convex/_generated/api";
import { useRouter } from "expo-router";
import Animated, { FadeIn, FadeInDown } from "react-native-reanimated";
import * as Haptics from "expo-haptics";
import { Platform, Switch } from "react-native";
import type { LocationBucket } from "../../types";

const LOCATION_OPTIONS: { value: LocationBucket; label: string; icon: string }[] = [
  { value: "home", label: "Home", icon: "🏠" },
  { value: "gym", label: "Gym", icon: "💪" },
  { value: "transit", label: "Transit", icon: "🚇" },
  { value: "other", label: "Other", icon: "📍" },
];

function SectionLabel({ title }: { title: string }) {
  return (
    <Text className="text-text-3 text-[11px] font-semibold uppercase tracking-widest px-1">
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
      disabled={!onPress}
      className="flex-row items-center gap-3 px-4 py-3.5"
    >
      <View className="w-9 h-9 rounded-xl bg-surface-3 items-center justify-center shrink-0">
        <Text className="text-[15px]">{icon}</Text>
      </View>
      <View className="flex-1 gap-0.5">
        <Text
          className="text-sm font-medium"
          style={{ color: danger ? "#ff453a" : "#ffffff" }}
        >
          {label}
        </Text>
        {sublabel && <Text className="text-text-3 text-xs">{sublabel}</Text>}
      </View>
      {right ?? (onPress && !danger ? <Text className="text-text-3 text-sm">›</Text> : null)}
    </Pressable>
  );
}

function CardGroup({ children }: { children: React.ReactNode }) {
  return (
    <View
      className="rounded-2xl bg-surface-2 overflow-hidden"
      style={{ borderWidth: 1, borderColor: "rgba(255,255,255,0.06)", borderCurve: "continuous" }}
    >
      {children}
    </View>
  );
}

export default function SettingsScreen() {
  const router = useRouter();
  const { sessionToken, logout } = useAuth();
  const [locationOverride, setLocationOverride] = useState<LocationBucket | null>(null);
  const [weatherEnabled, setWeatherEnabled] = useState(false);

  const user = useQuery(api.queries.getMe, sessionToken ? { sessionToken } : "skip");
  const weights = useQuery(api.queries.getUserWeights, sessionToken ? { sessionToken } : "skip");

  const handleLocationSelect = useCallback(async (value: LocationBucket) => {
    if (Platform.OS === "ios") await Haptics.selectionAsync();
    setLocationOverride((prev) => (prev === value ? null : value));
    // TODO: persist override to Convex
  }, []);

  const handleWeatherToggle = useCallback((value: boolean) => {
    setWeatherEnabled(value);
    // TODO: persist weatherEnabled to Convex
  }, []);

  const handleLogout = useCallback(async () => {
    if (Platform.OS === "ios") {
      await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
    }
    await logout();
    router.replace("/login");
  }, [logout, router]);

  return (
    <View className="flex-1 bg-surface-0">
      <ScrollView
        contentInsetAdjustmentBehavior="automatic"
        contentContainerClassName="px-5 pt-16 pb-32 gap-6"
      >
        {/* Title */}
        <Animated.View entering={FadeIn.delay(40).springify().damping(14)} className="px-1">
          <Text className="text-text-1 font-bold text-3xl tracking-tight">Settings</Text>
        </Animated.View>

        {/* Account */}
        <Animated.View entering={FadeInDown.delay(90).springify().damping(16)} className="gap-2.5">
          <SectionLabel title="Account" />
          <CardGroup>
            {user === undefined ? (
              <View className="px-4 py-4 gap-2">
                <Skeleton style={{ height: 16, width: "60%", borderRadius: 8 }} />
                <Skeleton style={{ height: 12, width: "40%", borderRadius: 6 }} />
              </View>
            ) : (
              <View className="px-4 py-4 flex-row items-center gap-3">
                <View className="w-11 h-11 rounded-full bg-surface-3 items-center justify-center overflow-hidden">
                  {user?.avatarUrl ? (
                    <Image
                      source={{ uri: user.avatarUrl }}
                      style={{ width: 44, height: 44 }}
                      contentFit="cover"
                    />
                  ) : (
                    <Text className="text-lg">🎵</Text>
                  )}
                </View>
                <View className="flex-1 gap-0.5">
                  <Text className="text-text-1 text-sm font-semibold">
                    {user?.displayName ?? "—"}
                  </Text>
                  <View className="flex-row items-center gap-1.5">
                    <View className="w-1.5 h-1.5 rounded-full bg-attune" />
                    <Text className="text-text-3 text-xs">Spotify connected</Text>
                  </View>
                </View>
              </View>
            )}
          </CardGroup>
        </Animated.View>

        {/* Model status */}
        <Animated.View entering={FadeInDown.delay(140).springify().damping(16)} className="gap-2.5">
          <SectionLabel title="Your Model" />
          <CardGroup>
            <SettingsRow
              icon="🧠"
              label="Personal taste model"
              sublabel={
                weights
                  ? `Trained on ${weights.eventCount} events · v${weights.modelVersion}`
                  : user?.onboardingStatus === "ready"
                  ? "Ready"
                  : "Not trained yet"
              }
            />
          </CardGroup>
        </Animated.View>

        {/* Location */}
        <Animated.View entering={FadeInDown.delay(190).springify().damping(16)} className="gap-2.5">
          <SectionLabel title="Location" />
          <CardGroup>
            <View className="px-4 py-4 gap-3">
              <View className="gap-0.5">
                <Text className="text-text-1 text-sm font-medium">Override location</Text>
                <Text className="text-text-3 text-xs">
                  Manually set where you are for better picks
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
                          ? "rgba(0,232,122,0.14)"
                          : "rgba(255,255,255,0.06)",
                      borderWidth: 1,
                      borderColor:
                        locationOverride === value
                          ? "rgba(0,232,122,0.38)"
                          : "rgba(255,255,255,0.08)",
                    }}
                  >
                    <Text className="text-sm">{icon}</Text>
                    <Text
                      className="text-xs font-medium"
                      style={{
                        color:
                          locationOverride === value ? "#00e87a" : "rgba(255,255,255,0.65)",
                      }}
                    >
                      {label}
                    </Text>
                  </Pressable>
                ))}
              </View>
            </View>
          </CardGroup>
        </Animated.View>

        {/* Personalization */}
        <Animated.View entering={FadeInDown.delay(240).springify().damping(16)} className="gap-2.5">
          <SectionLabel title="Personalization" />
          <CardGroup>
            <SettingsRow
              icon="🌤"
              label="Weather-aware picks"
              sublabel="Tunes energy to your local weather"
              right={
                <Switch
                  value={weatherEnabled}
                  onValueChange={handleWeatherToggle}
                  trackColor={{ false: "#333", true: "#00e87a" }}
                  thumbColor="#fff"
                />
              }
            />
          </CardGroup>
        </Animated.View>

        {/* Privacy */}
        <Animated.View entering={FadeInDown.delay(290).springify().damping(16)} className="gap-2.5">
          <SectionLabel title="Privacy" />
          <CardGroup>
            <View className="px-4 py-4 gap-2">
              <Text className="text-text-1 text-sm font-medium">Your data stays yours</Text>
              <Text className="text-text-3 text-xs leading-relaxed">
                Your personal model lives in your private Convex deployment. No raw GPS — only
                coarse location buckets (home, gym, transit). You can delete all data at any time.
              </Text>
            </View>
          </CardGroup>
        </Animated.View>

        {/* Danger zone */}
        <Animated.View entering={FadeInDown.delay(340).springify().damping(16)} className="gap-2.5">
          <SectionLabel title="Danger Zone" />
          <CardGroup>
            <SettingsRow
              icon="🚪"
              label="Disconnect Spotify"
              sublabel="Clears your session — model stays saved"
              danger
              onPress={handleLogout}
            />
          </CardGroup>
        </Animated.View>

        <Animated.View entering={FadeInDown.delay(390).springify().damping(18)}>
          <Text className="text-text-3 text-xs text-center">Attune v1.0.0</Text>
        </Animated.View>
      </ScrollView>
    </View>
  );
}

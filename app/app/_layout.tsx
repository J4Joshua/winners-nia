import "../src/global.css";
import { useEffect } from "react";
import { ConvexProvider } from "convex/react";
import { convex } from "../lib/convex";
import { Stack, useRouter, useSegments } from "expo-router";
import { StatusBar } from "expo-status-bar";
import { GestureHandlerRootView } from "react-native-gesture-handler";
import { AuthProvider, useAuth } from "../contexts/AuthContext";
import { PlaybackProvider } from "../contexts/PlaybackContext";
import { useQuery } from "convex/react";
import { api } from "../convex/_generated/api";
import Animated, { FadeIn } from "react-native-reanimated";
import { View, Text } from "../src/tw";

function SplashScreen() {
  return (
    <View className="flex-1 bg-surface-0 items-center justify-center">
      <Animated.View entering={FadeIn.duration(400)}>
        <View
          className="w-16 h-16 rounded-full bg-attune items-center justify-center"
          style={{ boxShadow: "0 0 32px rgba(0,232,122,0.3)" }}
        >
          <Text style={{ fontSize: 28, lineHeight: 32 }}>♪</Text>
        </View>
      </Animated.View>
    </View>
  );
}

function AuthGate({ children }: { children: React.ReactNode }) {
  const { sessionToken, isLoading, logout } = useAuth();
  const segments = useSegments();
  const router = useRouter();

  const me = useQuery(
    api.queries.getMe,
    sessionToken ? { sessionToken } : "skip"
  );

  useEffect(() => {
    if (isLoading) return;

    const inTabs = segments[0] === "(tabs)";
    const inLogin = segments[0] === "login";
    const inOnboarding = segments[0] === "onboarding";

    if (!sessionToken) {
      if (!inLogin) router.replace("/login");
      return;
    }

    // Session exists but user query returned (not still loading)
    if (me === undefined) return;

    // Session token is stale — the user record was deleted server-side
    if (me === null) {
      logout().then(() => router.replace("/login"));
      return;
    }

    if (me.onboardingStatus === "pending" && !inOnboarding) {
      router.replace("/onboarding");
      return;
    }

    if (me.onboardingStatus !== "pending" && !inTabs && !inOnboarding) {
      router.replace("/(tabs)");
    }
  }, [sessionToken, isLoading, me, segments, logout]);

  if (isLoading) return <SplashScreen />;

  return <>{children}</>;
}

function RootLayoutInner() {
  return (
    <AuthGate>
      <Stack screenOptions={{ headerShown: false, animation: "ios" }}>
        <Stack.Screen name="(tabs)" />
        <Stack.Screen name="login" options={{ animation: "fade" }} />
        <Stack.Screen name="onboarding" options={{ animation: "slide_from_bottom" }} />
      </Stack>
    </AuthGate>
  );
}

export default function RootLayout() {
  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <ConvexProvider client={convex}>
        <AuthProvider>
          <PlaybackProvider>
            <StatusBar style="light" />
            <RootLayoutInner />
          </PlaybackProvider>
        </AuthProvider>
      </ConvexProvider>
    </GestureHandlerRootView>
  );
}

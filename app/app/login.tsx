import { useEffect, useState } from "react";
import { useRouter } from "expo-router";
import { View, Text, Pressable } from "../src/tw";
import { useAuth } from "../contexts/AuthContext";
import { useSpotifyAuthRequest } from "../lib/spotify";
import { useAction } from "convex/react";
import { api } from "../convex/_generated/api";
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  withRepeat,
  withTiming,
  FadeIn,
  FadeInDown,
  Easing,
} from "react-native-reanimated";
import * as Haptics from "expo-haptics";
import { Platform } from "react-native";

function PulseRing({
  scale: scaleTarget,
  duration,
}: {
  scale: number;
  duration: number;
}) {
  const s = useSharedValue(0.85);
  const op = useSharedValue(0.45);

  useEffect(() => {
    s.value = withRepeat(
      withTiming(scaleTarget, { duration, easing: Easing.out(Easing.quad) }),
      -1,
      true
    );
    op.value = withRepeat(
      withTiming(0, { duration }),
      -1,
      true
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scaleTarget, duration]);

  const style = useAnimatedStyle(() => ({
    transform: [{ scale: s.value }],
    opacity: op.value,
  }));

  return (
    <Animated.View
      style={[
        style,
        {
          position: "absolute",
          width: 96,
          height: 96,
          borderRadius: 48,
          backgroundColor: "#00e87a",
        },
      ]}
    />
  );
}

export default function LoginScreen() {
  const router = useRouter();
  const { sessionToken, setSession } = useAuth();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { request, promptAsync, redirectUri } = useSpotifyAuthRequest();
  const linkSpotify = useAction(api.spotify.link);

  const buttonScale = useSharedValue(1);
  const buttonStyle = useAnimatedStyle(() => ({
    transform: [{ scale: buttonScale.value }],
  }));

  useEffect(() => {
    if (sessionToken) router.replace("/(tabs)");
  }, [sessionToken, router]);

  const handleLogin = async () => {
    if (!request || loading) return;
    if (Platform.OS === "ios") {
      await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    }

    buttonScale.value = withSpring(0.95, { damping: 8, stiffness: 350 }, () => {
      buttonScale.value = withSpring(1, { damping: 12, stiffness: 300 });
    });

    setLoading(true);
    setError(null);

    try {
      const result = await promptAsync();

      if (result.type === "cancel") return;
      if (result.type !== "success") {
        setError("Authentication failed — please try again");
        return;
      }

      const codeVerifier = request.codeVerifier;
      if (!codeVerifier) throw new Error("Missing PKCE code verifier");

      const { sessionToken: token } = await linkSpotify({
        code: result.params.code,
        redirectUri,
        codeVerifier,
      });

      await setSession(token);
      // Auth gate in _layout.tsx will handle redirect once session is set
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong — please try again");
    } finally {
      setLoading(false);
    }
  };

  return (
    <View className="flex-1 bg-surface-0">
      <View
        style={{
          position: "absolute",
          top: "18%",
          alignSelf: "center",
          width: 380,
          height: 380,
          borderRadius: 190,
          backgroundColor: "rgba(0, 232, 122, 0.05)",
        }}
      />

      <View className="flex-1 px-7 justify-end pb-14 gap-10">
        {/* Hero */}
        <Animated.View
          entering={FadeIn.delay(80).springify().damping(14)}
          className="items-center gap-7"
        >
          <View style={{ width: 96, height: 96, alignItems: "center", justifyContent: "center" }}>
            <PulseRing scale={1.85} duration={1800} />
            <PulseRing scale={1.5} duration={1400} />
            <View
              className="w-[84px] h-[84px] rounded-full bg-attune items-center justify-center"
              style={{ boxShadow: "0 0 40px rgba(0,232,122,0.35)" }}
            >
              <Text style={{ fontSize: 36, lineHeight: 40 }}>♪</Text>
            </View>
          </View>

          <View className="items-center gap-2">
            <Text className="text-text-1 font-bold text-[42px] tracking-tight leading-tight">
              Attune
            </Text>
            <Text className="text-text-2 text-base text-center leading-relaxed">
              Music shaped by{"\n"}this exact moment
            </Text>
          </View>
        </Animated.View>

        {/* Feature list */}
        <Animated.View entering={FadeInDown.delay(200).springify().damping(16)} className="gap-2">
          {[
            { icon: "🎯", label: "Learns from every skip & replay" },
            { icon: "📍", label: "Adapts to time, place & mood" },
            { icon: "⚡", label: "Gets sharper with every session" },
          ].map(({ icon, label }, i) => (
            <Animated.View
              key={label}
              entering={FadeInDown.delay(220 + i * 55).springify().damping(18)}
            >
              <View
                className="flex-row items-center gap-3 px-4 py-3 rounded-2xl"
                style={{
                  backgroundColor: "rgba(255,255,255,0.05)",
                  borderWidth: 1,
                  borderColor: "rgba(255,255,255,0.07)",
                }}
              >
                <View className="w-8 h-8 rounded-xl bg-surface-3 items-center justify-center flex-shrink-0">
                  <Text className="text-base">{icon}</Text>
                </View>
                <Text className="text-text-2 text-sm">{label}</Text>
              </View>
            </Animated.View>
          ))}
        </Animated.View>

        {/* CTA */}
        <Animated.View
          entering={FadeInDown.delay(400).springify().damping(16)}
          className="gap-4"
        >
          <Animated.View style={buttonStyle}>
            <Pressable
              onPress={handleLogin}
              disabled={loading}
              className="h-[56px] rounded-2xl items-center justify-center"
              style={{
                backgroundColor: loading ? "#00b85f" : "#00e87a",
                boxShadow: loading ? undefined : "0 0 24px rgba(0,232,122,0.28)",
              }}
            >
              <Text className="font-bold text-[15px] tracking-tight" style={{ color: "#000" }}>
                {loading ? "Connecting to Spotify…" : "Continue with Spotify"}
              </Text>
            </Pressable>
          </Animated.View>

          {error ? (
            <Text className="text-sf-red text-sm text-center">{error}</Text>
          ) : null}

          <Text className="text-text-3 text-xs text-center leading-relaxed">
            Your listening model lives in your private deployment.{"\n"}
            No raw GPS is ever stored. Delete all data anytime.
          </Text>
        </Animated.View>
      </View>
    </View>
  );
}

import { Tabs } from "expo-router";
import { Platform, PlatformColor } from "react-native";

const TINT = "#00e87a";
const INACTIVE = "rgba(255,255,255,0.4)";
const BG = Platform.OS === "ios" ? "transparent" : "#111111";

export default function TabsLayout() {
  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarActiveTintColor: TINT,
        tabBarInactiveTintColor: INACTIVE,
        tabBarStyle: {
          backgroundColor: BG,
          borderTopColor: "rgba(255,255,255,0.08)",
          borderTopWidth: 0.5,
        },
        tabBarLabelStyle: {
          fontSize: 10,
          fontFamily: Platform.OS === "ios" ? "ui-rounded" : undefined,
        },
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: "Home",
          tabBarIcon: ({ color, size }) => (
            <TabIcon glyph="⊙" color={color} size={size} />
          ),
        }}
      />
      <Tabs.Screen
        name="player"
        options={{
          title: "Now Playing",
          tabBarIcon: ({ color, size }) => (
            <TabIcon glyph="♫" color={color} size={size} />
          ),
        }}
      />
      <Tabs.Screen
        name="settings"
        options={{
          title: "Settings",
          tabBarIcon: ({ color, size }) => (
            <TabIcon glyph="⚙" color={color} size={size} />
          ),
        }}
      />
    </Tabs>
  );
}

function TabIcon({ glyph, color, size }: { glyph: string; color: string; size: number }) {
  const { Text } = require("react-native");
  return <Text style={{ fontSize: size - 4, color }}>{glyph}</Text>;
}

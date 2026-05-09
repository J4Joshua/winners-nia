import { Tabs } from "expo-router";
import { Image } from "expo-image";
import { Text } from "react-native";
import { Platform } from "react-native";

const TINT = "#00e87a";
const INACTIVE = "rgba(255,255,255,0.38)";

interface TabIconProps {
  sfSymbol: string;
  fallback: string;
  color: string;
  focused: boolean;
}

function TabIcon({ sfSymbol, fallback, color }: TabIconProps) {
  if (Platform.OS === "ios") {
    return (
      <Image
        source={`sf:${sfSymbol}`}
        style={{ width: 24, height: 24 }}
        tintColor={color}
        contentFit="contain"
      />
    );
  }
  return <Text style={{ fontSize: 20, color }}>{fallback}</Text>;
}

export default function TabsLayout() {
  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarActiveTintColor: TINT,
        tabBarInactiveTintColor: INACTIVE,
        tabBarStyle: {
          backgroundColor:
            Platform.OS === "ios" ? "rgba(10,10,10,0.94)" : "#111111",
          borderTopColor: "rgba(255,255,255,0.08)",
          borderTopWidth: 0.5,
        },
        tabBarLabelStyle: {
          fontSize: 10,
          fontFamily: Platform.OS === "ios" ? "ui-rounded" : undefined,
          fontWeight: "500",
        },
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: "Home",
          tabBarIcon: (props) => (
            <TabIcon {...props} sfSymbol="house.fill" fallback="⌂" />
          ),
        }}
      />
      <Tabs.Screen
        name="player"
        options={{
          title: "Now Playing",
          tabBarIcon: (props) => (
            <TabIcon {...props} sfSymbol="music.note" fallback="♫" />
          ),
        }}
      />
      <Tabs.Screen
        name="settings"
        options={{
          title: "Settings",
          tabBarIcon: (props) => (
            <TabIcon {...props} sfSymbol="gearshape.fill" fallback="⚙" />
          ),
        }}
      />
    </Tabs>
  );
}

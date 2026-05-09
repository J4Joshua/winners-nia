import { ConvexReactClient } from "convex/react";
import Constants from "expo-constants";

const convexUrl =
  process.env.EXPO_PUBLIC_CONVEX_URL ??
  (Constants.expoConfig?.extra?.convexUrl as string | undefined);

if (!convexUrl) {
  throw new Error(
    "EXPO_PUBLIC_CONVEX_URL is not set. Run `npx convex dev` and ensure .env.local is configured."
  );
}

export const convex = new ConvexReactClient(convexUrl);

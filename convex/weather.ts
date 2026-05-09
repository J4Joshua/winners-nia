"use node";

import { action } from "./_generated/server";
import { v } from "convex/values";

type WeatherCondition = "clear" | "clouds" | "rain" | "snow" | "storm";

function mapCondition(main: string): WeatherCondition {
  const lower = main.toLowerCase();
  if (lower.includes("clear") || lower.includes("sunny")) return "clear";
  if (lower.includes("cloud") || lower.includes("mist") || lower.includes("fog")) return "clouds";
  if (lower.includes("rain") || lower.includes("drizzle")) return "rain";
  if (lower.includes("snow") || lower.includes("sleet")) return "snow";
  if (lower.includes("thunder") || lower.includes("storm")) return "storm";
  return "clear";
}

export const getWeather = action({
  args: { lat: v.number(), lng: v.number() },
  handler: async (_ctx, { lat, lng }) => {
    const apiKey = process.env.OPENWEATHER_API_KEY;
    if (!apiKey) {
      // Return a neutral fallback if not configured
      return { temp: 20, condition: "clear" as WeatherCondition, humidity: 50, isDay: true };
    }

    const url = `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lng}&appid=${apiKey}&units=metric`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`OpenWeather error: ${res.status}`);

    const data = (await res.json()) as {
      main: { temp: number; humidity: number };
      weather: { main: string }[];
      sys: { sunrise: number; sunset: number };
      dt: number;
    };

    const isDay = data.dt >= data.sys.sunrise && data.dt <= data.sys.sunset;

    return {
      temp: Math.round(data.main.temp),
      condition: mapCondition(data.weather[0]?.main ?? ""),
      humidity: data.main.humidity,
      isDay,
    };
  },
});

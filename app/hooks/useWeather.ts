import { useCallback, useState } from "react";
import { useAction } from "convex/react";
import { api } from "../convex/_generated/api";
import type { WeatherCondition } from "./useTelemetry";

export interface WeatherData {
  temp: number;
  condition: WeatherCondition;
  humidity: number;
  isDay: boolean;
}

export function useWeather() {
  const [weather, setWeather] = useState<WeatherData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const getWeatherAction = useAction(api.weather.getWeather);

  const fetchWeather = useCallback(
    async (lat: number, lng: number) => {
      setLoading(true);
      setError(null);
      try {
        const result = await getWeatherAction({ lat, lng });
        setWeather(result as WeatherData);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Weather fetch failed");
      } finally {
        setLoading(false);
      }
    },
    [getWeatherAction]
  );

  return { weather, loading, error, fetchWeather };
}

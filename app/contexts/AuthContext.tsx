import React, { createContext, useCallback, useContext, useEffect, useState } from "react";
import AsyncStorage from "@react-native-async-storage/async-storage";

const SESSION_KEY = "@attune/sessionToken";

interface AuthContextValue {
  sessionToken: string | null;
  isLoading: boolean;
  setSession: (token: string | null) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [sessionToken, setSessionTokenState] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    AsyncStorage.getItem(SESSION_KEY)
      .then((token) => {
        if (token) setSessionTokenState(token);
      })
      .finally(() => setIsLoading(false));
  }, []);

  const setSession = useCallback(async (token: string | null) => {
    if (token) {
      await AsyncStorage.setItem(SESSION_KEY, token);
    } else {
      await AsyncStorage.removeItem(SESSION_KEY);
    }
    setSessionTokenState(token);
  }, []);

  const logout = useCallback(async () => {
    await AsyncStorage.removeItem(SESSION_KEY);
    setSessionTokenState(null);
  }, []);

  return (
    <AuthContext.Provider value={{ sessionToken, isLoading, setSession, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}

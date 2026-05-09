import { internalMutation, mutation } from "./_generated/server";
import { v } from "convex/values";
import { requireSession } from "./lib/auth";

export const startOnboarding = mutation({
  args: { sessionToken: v.string() },
  handler: async (ctx, { sessionToken }) => {
    const user = await requireSession(ctx, sessionToken);

    if (user.onboardingStatus !== "pending") {
      return { alreadyStarted: true };
    }

    const existing = await ctx.db
      .query("onboardingJobs")
      .withIndex("by_user", (q) => q.eq("userId", user._id))
      .first();

    if (existing) {
      await ctx.db.patch(existing._id, {
        status: "fetching",
        errorMessage: undefined,
        startedAt: Date.now(),
        completedAt: undefined,
      });
    } else {
      await ctx.db.insert("onboardingJobs", {
        userId: user._id,
        status: "fetching",
        startedAt: Date.now(),
      });
    }

    await ctx.db.patch(user._id, { onboardingStatus: "fetching" });

    // TODO: schedule the Spotify data fetch + cold train action
    // await ctx.scheduler.runAfter(0, internal.onboarding.fetchAndTrain, { userId: user._id });

    return { alreadyStarted: false };
  },
});

export const updateJob = internalMutation({
  args: {
    userId: v.id("users"),
    status: v.union(
      v.literal("fetching"),
      v.literal("analyzing"),
      v.literal("training"),
      v.literal("ready"),
      v.literal("error")
    ),
    errorMessage: v.optional(v.string()),
    tracksFound: v.optional(v.number()),
  },
  handler: async (ctx, { userId, status, errorMessage, tracksFound }) => {
    const job = await ctx.db
      .query("onboardingJobs")
      .withIndex("by_user", (q) => q.eq("userId", userId))
      .first();

    if (!job) throw new Error("Onboarding job not found");

    await ctx.db.patch(job._id, {
      status,
      errorMessage,
      tracksFound,
      ...(status === "ready" || status === "error" ? { completedAt: Date.now() } : {}),
    });

    // Keep user doc in sync for all terminal and progress states
    if (status === "ready") {
      await ctx.db.patch(userId, { onboardingStatus: "ready" });
    } else if (status === "error") {
      await ctx.db.patch(userId, { onboardingStatus: "pending" });
    } else {
      await ctx.db.patch(userId, { onboardingStatus: status as "fetching" });
    }
  },
});

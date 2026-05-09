import type { MutationCtx, QueryCtx } from "../_generated/server";
import type { Doc } from "../_generated/dataModel";

/**
 * Validate a session token and return the authenticated user.
 * Throws "Unauthorized" if the token is missing or invalid.
 *
 * Safe to call from queries and mutations (no Date.now() — sessions are
 * long-lived and invalidated explicitly via deletion).
 */
export async function requireSession(
  ctx: QueryCtx | MutationCtx,
  sessionToken: string
): Promise<Doc<"users">> {
  const session = await ctx.db
    .query("sessions")
    .withIndex("by_token", (q) => q.eq("token", sessionToken))
    .unique();

  if (!session) throw new Error("Unauthorized");

  const user = await ctx.db.get(session.userId);
  if (!user) throw new Error("User not found");

  return user;
}

# ADR 0001 — Stack (Attune, hackathon)

**Status:** Accepted  
**Date:** 2026-05-09  
**Track:** **Ship It** (public demo; not optimizing Always-on / Nia unless scope changes)

## Context

Hackathon build with real **model training** (Song / User / Context / GRU per `ml/README.md`), public demo, Spotify playback. Inference on **RunPod Serverless** — see [`runpod-serverless.md`](../runpod-serverless.md).

## Decision

| Layer | Choice |
|-------|--------|
| Client | **Expo** — **iOS**, **Android**, and **Web** (same codebase; web uses Web Playback SDK where native Remote is unavailable) |
| Auth | **Spotify login from the client** (PKCE / `expo-auth-session`); after success, **exchange code server-side in Convex** and persist refresh (**B**) |
| Backend | **Convex** — users, encrypted Spotify refresh, events, playlist orchestration; **actions** call RunPod with `RUNPOD_API_KEY` |
| Inference | **RunPod Serverless** — queue endpoint, Python **handler**, `POST .../runsync` (or `/run` + `/status`); Convex holds API key — [`runpod-serverless.md`](../runpod-serverless.md) |
| GPU train | **RunPod** Pods/Jobs — Song / GRU / heavy trains; export artifacts baked into Serverless worker image or volume |
| Context v1 | **Weather**, **time**, **coarse location**, **playback** on events + rank `input` |
| Spotify tokens | **B** — refresh in Convex only; short-lived access token passed to RunPod **inside** `input` when the worker must call Spotify; never ship refresh to Expo or RunPod env |

## Inference split

- **Convex:** mutations (events), queries, internal token refresh, **actions** that `fetch` RunPod `runsync` / `health`.
- **RunPod Serverless worker:** handler implements `rank` (and optionally `online_step`); loads FAISS + torch from image/volume.

## Spotify (client login, server trust)

1. Expo opens Spotify auth (PKCE) for **iOS / Android / Web** (web redirect URIs must match Spotify dashboard).
2. Expo sends **authorization `code`** to Convex mutation/action endpoint.
3. Convex exchanges code, stores **refresh**, returns session identity to app.
4. Playlist / onboarding: Convex action refreshes Spotify access as needed, builds `input` for RunPod (include access token in `input` only for that request).

## Deploy (Ship It)

- **Convex:** `npx convex deploy`; env `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`, Spotify client id/secret.
- **Expo:** **EAS** builds for iOS + Android; **web** deploy (EAS Hosting or static) with same Convex URL.
- **RunPod:** create Serverless endpoint from worker image; paste **endpoint id** into Convex.

## Out of scope

- Convex-hosted PyTorch inference.
- Always-on agent track unless you explicitly re-open scope.

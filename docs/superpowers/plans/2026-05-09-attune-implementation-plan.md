# Attune — Implementation Plan

> **For agentic workers:** Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to execute task-by-task. Steps use `- [ ]` syntax for tracking.

---

## Overview

| | |
|---|---|
| **What** | Personal music remixer that learns from *this* user's behavior, combining Spotify playback with context-aware ML |
| **Stack** | Expo (iOS/Android/Web) · Convex · RunPod (GPU train + Serverless inference) |
| **Track** | Hackathon **Ship It** — public demo, real model training |
| **Auth** | Spotify PKCE on client → code exchange in Convex → refresh stored server-side (**B**) |
| **Repo** | Monorepo: Python ML package + Convex backend + Expo app + RunPod worker |

---

## Part A — Architecture & models

### How it works (plain English)

Think of it like this: every song can be described as a **point in space**. Two songs that "feel" similar (same vibe, energy, tempo) end up close together in that space. We also put the **user** as a point in that same space — the user's point is near the songs they like.

To pick the next song, we just look for **songs near the user's current point**. The trick is that the user's point **moves** based on what's happening *right now* — time of day, location, mood (inferred from skipping), weather — and based on what they just listened to in this session.

We have four small "brains" (neural networks) that each handle one job:

| Brain | What it does (analogy) | Plain explanation |
|-------|------------------------|-------------------|
| **Song Brain** | The **librarian** who files every song in the right shelf | Takes 26 numbers that describe a song (tempo, energy, danceability, etc.) and converts them into a **position** in our imaginary space. Trained once on ~114k songs, then never changes. |
| **User Brain** | The **personal stylist** that knows your taste | Takes 17 numbers summarizing your listening habits and places *you* in the same space as the songs. **Each user gets their own copy** that updates every time you skip or replay. |
| **Context Brain** | The **mood ring** that adjusts for "right now" | Nudges your position based on 16 situational signals — what time it is, where you are (gym vs home), weather, how many skips in a row. It learns that "gym + morning" means higher energy. |
| **Session Brain** | The **DJ** who watches the current set | Looks at the sequence of songs you've played in *this* session and predicts where the next song should land. Prevents jarring transitions. |

### How picking a playlist works (step by step)

```
1. "Where is this user right now?" → User Brain places them in space
2. "What's happening around them?" → Context Brain nudges the position
3. "What songs are near that spot?" → Search the catalog (~114k songs), grab ~500 closest
4. "What have they been listening to?" → Session Brain refines the direction
5. "Which of those 500 score best?" → Rank by closeness to session direction
6. "Mix familiar + new" → Pick ~20 songs: some the user already knows, some discoveries
```

### How it learns from you (in real time)

Every time you interact — skip, replay, let a song finish — we use that signal to **update your User Brain** slightly. It's like nudging your point in space:
- You replayed a song → move toward it
- You skipped instantly → move away from it
- You're skipping a lot → Context Brain kicks in harder (more "safety" picks)

We keep a memory of your last ~200 interactions so we don't "forget" old preferences while chasing new ones. This prevents the system from overcorrecting.

### The four models (technical reference)

| Model | Architecture | Input → Output | Training |
|-------|-------------|----------------|----------|
| **Song Tower** | MLP: Linear(26,256) → ReLU → Linear(256,128) → L2-norm | `(batch, 26)` → `(batch, 128)` | InfoNCE with in-batch negatives on ~114k [HF tracks](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset). Single GPU session (~20–40 min on 4090). Frozen after. |
| **User Tower** | MLP: Linear(17,256) → ReLU → Linear(256,128) → L2-norm | `(batch, 17)` → `(batch, 128)` | Cold: fit from Spotify history (~5–15 min CPU). Online: 1 Adam step per event (lr 1e-5, grad clip 0.3, batch = 1 new + 8 replayed from 200-event buffer). |
| **Context Encoder** | Gated residual: `user_vec + gate * shift` where gate = σ(Linear(context)), shift = Linear(context) | User 128-d + context 16-d → shifted 128-d | Offline on (context, user_vec, outcome_song) triplets. Synthetic for hackathon. Weather gated off until ~2–3 weeks data. |
| **GRU Sequencer** | GRU: input 141-d, hidden 256-d → Linear(256,128) → L2-norm | `(seq_len, 141)` → `(128,)` query | Offline teacher-forcing on session-ordered events. Synthetic for hackathon. |

### 26-d song feature vector (`features_song_v1`)

**Plain English:** Each song is described by 26 numbers. 12 of them say what musical key the song is in (like "C major" or "F# minor" — we turn this into a bunch of yes/no flags). The other 14 are things like how danceable it is, how energetic, how loud, how fast, etc. All numbers are squished to roughly 0–1 so they're on the same scale.

**Structure:** 12 key one-hot + 14 scalars = 26 dimensions.

| Index | Feature | Transform |
|-------|---------|-----------|
| 0–11 | `key` one-hot | If `key` in `[0..11]` → 1.0 at that index; if missing/-1 → all zeros |
| 12 | `danceability` | Raw (already 0–1) |
| 13 | `energy` | Raw (already 0–1) |
| 14 | `speechiness` | Raw (already 0–1) |
| 15 | `acousticness` | Raw (already 0–1) |
| 16 | `instrumentalness` | Raw (already 0–1) |
| 17 | `liveness` | Raw (already 0–1) |
| 18 | `valence` | Raw (already 0–1) |
| 19 | `norm_loudness` | `clamp((loudness + 60) / 60, 0, 1)` |
| 20 | `tempo_norm` | `tempo / 240` |
| 21 | `mode` | Raw (0 or 1) |
| 22 | `explicit` | 1.0 if true else 0.0 |
| 23 | `popularity_norm` | `popularity / 100` |
| 24 | `duration_norm` | `min(duration_ms / 330_000, 1.0)` |
| 25 | `time_signature_norm` | `clamp(time_signature / 7, 0, 1)` |

### 17-d user feature vector (`features_user_v1`)

**Plain English:** We describe each user with 17 numbers summarizing their taste — average danceability of their top songs, how diverse their genres are, how often they skip, what time of day they listen most, etc. This is the input to the User Brain.

| Index | Feature | Source |
|-------|---------|--------|
| 0 | `mean_danceability` | Mean of user's top-50 tracks |
| 1 | `mean_energy` | Same |
| 2 | `mean_valence` | Same |
| 3 | `mean_acousticness` | Same |
| 4 | `mean_instrumentalness` | Same |
| 5 | `mean_tempo_norm` | Mean tempo/240 |
| 6 | `mean_loudness_norm` | Mean normalized loudness |
| 7 | `genre_diversity` | Unique genres / total genres in top-50 |
| 8 | `listening_recency` | Fraction of recently-played in last 7 days |
| 9 | `skip_rate_30d` | Overall skip rate (or 0.5 if new) |
| 10 | `replay_rate_30d` | Fraction of tracks replayed |
| 11 | `avg_listen_ratio` | Mean listen ratio across recent plays |
| 12 | `discovery_ratio` | Fraction of unique new tracks / total listened |
| 13 | `session_length_avg` | Mean tracks per session |
| 14 | `peak_hour_sin` | sin(2π × most_active_hour / 24) |
| 15 | `peak_hour_cos` | cos(2π × most_active_hour / 24) |
| 16 | `account_age_norm` | min(days_since_first_event / 365, 1.0) (0 for new) |

### 16-d context vector (Context Encoder input)

**Plain English:** 16 numbers that describe "what's happening right now" — the time encoded as a cycle (so 11pm and 1am are close), day of week, how much the user is skipping in this session, where they are (gym/home/transit), and what the weather is like. This is the input to the Context Brain (the "mood ring").

| Index | Feature | Source |
|-------|---------|--------|
| 0 | `hour_sin` | sin(2π × hour_local / 24) |
| 1 | `hour_cos` | cos(2π × hour_local / 24) |
| 2 | `dow_sin` | sin(2π × dow / 7) |
| 3 | `dow_cos` | cos(2π × dow / 7) |
| 4 | `session_skip_rate` | Skips / total in current session so far |
| 5 | `consecutive_skips_norm` | min(consecutive_skips / 5, 1.0) |
| 6 | `session_length_norm` | min(tracks_in_session / 30, 1.0) |
| 7 | `volume_delta` | Recent volume change, normalized [-1, 1] |
| 8 | `location_home` | 1.0 if bucket == home |
| 9 | `location_gym` | 1.0 if bucket == gym |
| 10 | `location_transit` | 1.0 if bucket == transit |
| 11 | `location_other` | 1.0 if none of above |
| 12 | `weather_temp_norm` | (temp_celsius + 20) / 60, clamped [0,1] |
| 13 | `weather_is_rain` | 1.0 if rainy/stormy |
| 14 | `weather_humidity_norm` | humidity / 100 |
| 15 | `weather_is_day` | 1.0 if daytime |

### 141-d GRU step input

**Plain English:** For the DJ Brain (GRU), we feed it one song at a time as the session progresses. Each step is 141 numbers: the song's 128-d "position in space" plus 13 numbers about what happened (did they skip it? volume change? what time? where are they?). This lets the DJ Brain "watch" the session unfold.

128-d song embedding (from Song Tower for the track just heard) + 13 scalars:

| Offset | Feature |
|--------|---------|
| 0–127 | Song embedding of track just played |
| 128 | `listen_ratio` |
| 129 | `skip` (0/1) |
| 130 | `replay` (0/1) |
| 131 | `volume_delta` |
| 132 | `hour_sin` |
| 133 | `hour_cos` |
| 134 | `session_skip_rate` |
| 135 | `consecutive_skips_norm` |
| 136 | `location_home` |
| 137 | `location_gym` |
| 138 | `location_transit` |
| 139 | `location_other` |
| 140 | `position_in_session_norm` (idx / 30, capped 1.0) |

### Inference pipeline (per playlist request)

**In plain English:** When you tap "Attune this moment":
1. We figure out where *you* sit in song-space right now
2. We adjust that based on your current situation (gym? raining? late night?)
3. We look at the map of all ~114k songs and grab the 500 closest to your spot
4. We look at what you just listened to and fine-tune the direction ("they're on an upbeat streak")
5. We rank those 500 and pick the best ~20, mixing some you know (safe bets) with discoveries

**Technical pseudocode:**

```
1. user_vec = UserTower(user_features_17d)                   → (128,)
2. shifted_user = ContextEncoder(user_vec, context_16d)      → (128,)
3. candidates = FAISS.search(shifted_user, k=500)            → 500 track ids + scores
4. Split into known_pool / new_pool based on saved/recent
5. For each track in session_tail:
     gru_input = concat(song_embedding, event_scalars)       → (141,)
     gru_hidden = GRU.step(gru_input, gru_hidden)
6. session_query = Linear(gru_hidden)                        → (128,)
7. final_scores = dot(session_query, candidate_embeddings)   → (500,)
8. Interleave top known + top new → ~20 tracks
     Default: 2 known / 3 new
     High skip rate (>60%) → more familiar (3/2)
     Low skip rate (<20%) → more discovery (1/4)
     Gym → more familiar; Home night → more discovery
```

### How actions become training signals

**In plain English:** Every thing you do tells us something. We score each action:

| What you did | How much we trust it | What the system learns |
|--------------|---------------------|----------------------|
| Replayed a song | Very strong "love it" | Move your point strongly toward that song |
| Added to playlist | Almost certain you like it | Move toward it |
| Listened to >80% | Probably liked it | Gentle move toward it |
| Turned volume up | Enjoying it a bit more | Small bonus toward it |
| Listened 50–80% | It was okay | Tiny move toward it |
| Skipped after 5–30s | Not feeling it | Barely move (mild negative) |
| Skipped instantly (<5s) | Accident or hate | Ignore / move away slightly |
| 3 skips in a row | Something is off about *right now* | The system shifts strategy: more safe picks |

**Technical weights:** Replay=1.0, Add=0.95, Complete=0.85, VolUp=+0.15, Listen50–80%=0.60, EarlySkip=0.15, InstantSkip=0.0.

### Five improvement loops (how the system gets smarter)

| Loop | When it runs | What happens (plain) | Technical |
|------|-------------|---------------------|-----------|
| **Live learning** | After every song interaction | Your personal brain updates immediately — it scoots your point toward songs you liked, away from songs you hated | 1 Adam step, lr 1e-5, clip 0.3, 9-event batch (1 new + 8 replayed from memory). RunPod Serverless. |
| **Context tuning** | Weekly batch job | The "mood ring" brain learns better rules (e.g. "rain + evening → more acoustic") | Re-fit gate on accumulated data. RunPod GPU Pod. |
| **Session tuning** | After 1–2 weeks | The DJ brain gets better at transitions ("after 3 chill songs, maybe nudge energy up") | GRU teacher-forcing offline. RunPod GPU Pod. |
| **Mix policy** | Every request (instant) | Simple rules: "user is skipping a lot → play more safe songs they know" | No ML training — just if/else logic in the handler. |
| **New music** | Nightly (post-hackathon) | When new songs appear, put them on the map so they can be recommended | Run frozen Song Brain on new tracks → add to search index. |

### Why this approach works (research)

These aren't ideas we made up — they're proven patterns used by companies like Spotify, YouTube, and Netflix:

- **Two towers (Song Brain + User Brain):** This is how NVIDIA and others solve the "new user" problem — one brain for items, one for users, both placing things in the same space. [NVIDIA Merlin](https://medium.com/nvidia-merlin/solving-the-cold-start-problem-using-two-tower-neural-networks-for-nvidias-e-mail-recommender-2d5b30a071a4), [IEEE Access](https://ieeeaccess.ieee.org/featured-articles/flexible2-towermodel/)
- **Session-aware DJ:** Music is sequential — what works after a chill song is different from what works after an energetic one. Spotify Research and others model this with sequence models. [Spotify CoSeRNN](https://research.atspotify.com/contextual-and-sequential-user-embeddings-for-music-recommendation), [arXiv:1904.10273](https://arxiv.org/abs/1904.10273)
- **Learning without forgetting:** When you update a model on new data, it can "forget" old preferences. The fix: keep a memory (replay buffer) and mix old examples with new ones. This is standard practice. [AAAI INFER](https://ojs.aaai.org/index.php/AAAI/article/view/28790), [arXiv:2403.03993](https://arxiv.org/abs/2403.03993)

---

## Part B — App functionality & data gathering

### App features (all platforms)

| Feature | iOS/Android | Web |
|---------|-------------|-----|
| **Sign in** | PKCE via `expo-auth-session` → send code to Convex | Same PKCE; redirect URIs match Spotify dashboard + hosting domain |
| **Playback** | Spotify Remote SDK (`app-remote-control` scope) | [Web Playback SDK](https://developer.spotify.com/documentation/web-playback-sdk) (`streaming` scope, Premium required) |
| **"Attune this moment"** | One tap → collect context → Convex action → ~20 URIs → Spotify queue API | Same |
| **Telemetry** | Track change, progress every 5–15s, skip/replay/volume → batched to Convex with idempotency key | Same (volume may be unavailable on web — send null) |
| **Now playing** | Title, artist, album art, progress ring, skip/prev controls, volume indicator | Same |
| **Explainability v1** | Short rule-based string ("More familiar at the gym", "Pushed tempo up after skips") | Same |
| **Settings** | Unlink Spotify, privacy link, location bucket override (home/gym/transit/other) | Same |
| **Offline** | Queue events locally; flush on reconnect | Same |

### Spotify scopes required

```
user-read-recently-played
user-top-read
user-library-read
app-remote-control        (iOS/Android native playback)
streaming                 (Web Playback SDK — Premium only)
```

### Data collected

**Implicit feedback (always on after consent):**

| Signal | Trigger | Fields stored | Frequency |
|--------|---------|---------------|-----------|
| Progress | Timer + pause/stop/track-change | `position_ms`, `duration_ms`, `listen_ratio` | Every 5–15s while playing |
| Skip | User initiates next | `skip: true`, `skip_bucket` (instant <5s / early 5–30s / late >30s), `skip_position_ms` | Immediate |
| Replay | User restarts same track | `replay: true` | Immediate |
| Volume | OS volume callback (meaningful change ≥5%) | `volume_delta` normalized [-1, 1] | On change |
| Session stats | Computed per event | `session_skip_rate`, `consecutive_skips`, `session_idx` | Each event |
| Explicit like | User taps like/save (if `user-library-modify` added) | `explicit_like: true` | Immediate |

**Context snapshot (attached to each batch and each playlist request):**

| Field | Source | Notes |
|-------|--------|-------|
| `hour_local` (0–23) | Device clock | Integer |
| `dow` (0–6, Mon=0) | Device clock | Integer |
| `location_bucket` | GPS → reverse geocode → rule (home radius / gym radius / in-vehicle speed / other) OR manual override | Coarse — never store raw lat/lng |
| `weather.temp` | Convex action → OpenWeather/Open-Meteo (hides API key) | Celsius |
| `weather.condition` | Same | Enum: clear/clouds/rain/snow/storm |
| `weather.humidity` | Same | 0–100 |
| `weather.is_day` | Same | Boolean |
| `weather_enabled` | Convex flag | false until ≥21 days OR enough weather diversity |

**Spotify API pulls (Convex server-side with refresh token B):**

| Endpoint | Scope | Max items | Purpose |
|----------|-------|-----------|---------|
| `GET /me/player/recently-played` | `user-read-recently-played` | 50 | Short-term taste, recency |
| `GET /me/top/tracks?time_range=medium_term` | `user-top-read` | 50 | Medium-term preference |
| `GET /me/tracks` (saved) | `user-library-read` | Sample 100 | Known pool for interleave |
| `GET /audio-features?ids=...` | (no extra scope) | 100 per call | Song Tower features for those tracks |

**Rate limits:** Batch audio features calls (100 IDs per request). Cache results in Convex by `track_id` (never expires — audio features don't change).

### Onboarding (first ~15 min to first recommendation)

**Sequence diagram:**

```
User opens app
  │
  ├─ 1. Welcome screen
  │     "Playlists for this moment. Improves as you skip or stay."
  │
  ├─ 2. Spotify PKCE login (expo-auth-session)
  │     → Opens Spotify consent page
  │     → User approves scopes
  │     → Redirect back with authorization code
  │
  ├─ 3. Send code to Convex
  │     mutation: spotify.link({ code, redirectUri })
  │     Convex exchanges code → access + refresh tokens
  │     Store refresh (encrypted) in `users` table
  │     Create `onboardingJobs` doc: status = "fetching"
  │
  ├─ 4. Convex action: fetch Spotify data (background)
  │     GET /me/player/recently-played (50 tracks)
  │     GET /me/top/tracks?time_range=medium_term (50 tracks)
  │     GET /me/tracks (saved, sample 100)
  │     GET /audio-features for all unique track IDs (batch 100/call)
  │     → Compute features_song_v1 for each track
  │     → Compute features_user_v1 from aggregate stats
  │     → Build training pairs:
  │         positives = tracks with high listen ratio / saved / top
  │         negatives = random sample from global catalog (not in user history)
  │     Update onboardingJobs: status = "training"
  │
  ├─ 5. Cold User Tower train
  │     Option A: Inline in Convex action on CPU (~5–15 min)
  │     Option B: Start RunPod CPU Pod job (faster for batch)
  │     Train: InfoNCE or MSE loss, ~50–200 epochs on ~100–300 pairs
  │     Export: weights as base64 → store in `userWeights` table (~212 KB)
  │     Update onboardingJobs: status = "ready"
  │
  ├─ 6. App polls onboardingJobs or subscribes to Convex query
  │     When status = "ready" → show "You're ready!" + enable main button
  │
  ├─ 7. Permission prompts
  │     Location: "When In Use" (for bucket classification)
  │     Notifications: Optional ("daily session summary")
  │
  ├─ 8. First "Attune this moment" tap
  │     Collect context (time, location, weather if available)
  │     Session tail = empty (no history yet)
  │     Convex action → RunPod runsync: rank
  │     User Tower = cold weights, Context = maybe weather-gated-off, GRU = stub/bypass
  │     Return ~20 URIs → enqueue via Spotify
  │
  └─ 9. Start playing → events stream → online learning begins
        Each track change/skip/replay → appendBatch → online_step
        Model starts adapting within minutes
```

**Sparse history fallback:** If <10 tracks in Spotify history → use genre/top-artist biased candidates from global FAISS + set known fraction to 80% until ≥20 events.

---

## Part C — GPU compute plan

**Plain English:** Training the brains requires a powerful graphics card (GPU) — the same kind gamers use, but we rent them in the cloud from RunPod. Once the brains are trained, we package them into a Docker container and run them on RunPod Serverless — which means we only pay when someone actually asks for a playlist. Think of it like: training = the chef learning recipes (expensive, one-time), inference = the chef cooking an order (cheap, per-request).

### Three compute lanes

| Lane | RunPod product | Jobs | Billing |
|------|----------------|------|---------|
| **A. GPU Pod** | On-demand Pod with SSH | Song InfoNCE, embedding export, Context/GRU offline trains | Per-hour while running |
| **B. CPU** | CPU Pod / laptop / Convex action | HF → Parquet, FAISS build, User cold train, tests | Minimal |
| **C. Serverless** | Queue endpoint + Docker handler | `rank` and `online_step` inference | Per-second active (flex/active) |

### GPU/CPU per job (detailed)

| Job | Hardware | VRAM used | Wall-clock | Batch size | Notes | Cost |
|-----|----------|-----------|-----------|------------|-------|------|
| Song Tower InfoNCE | **RTX 4090 24GB** Pod | ~6–10 GB (AMP) | ~15–40 min | 512 (4090) / 256 (L4) | In-batch negatives; larger batch = better InfoNCE quality | ~$0.2–$0.6 |
| Song embeddings forward | Same Pod (chain after train) | ~2 GB | ~5–15 min | 4096–8192 | `torch.no_grad()`, AMP | marginal (same session) |
| FAISS index build | CPU 16GB RAM | N/A | ~1–5 min | N/A | `IndexFlatIP` on 114k × 128 float32 ≈ 58 MB vectors | free (laptop) |
| User Tower cold | CPU 8 vCPU | N/A | ~5–15 min | 64 | Small MLP; I/O-bound on feature prep; GPU optional for <5 min | ~$0 |
| Context Encoder | **4090** or **L4** Pod | ~4–8 GB | ~15–45 min (synthetic) | 256 | Synthetic data for hackathon; real data training takes longer | ~$0.2–$0.5 |
| GRU Sequencer | **4090** Pod | ~4–10 GB | ~10–30 min (synthetic) | 32 sessions × 20 steps | Teacher-forcing; variable session length | ~$0.2–$0.5 |
| Serverless `rank` | L4/4090 PRO 24GB | ~4 GB steady | 0.2–2s (warm) / 10–60s (cold) | 1 request | FAISS search + forward passes | ~$0.00019–$0.00031/s |
| Serverless `online_step` | Same endpoint | ~2 GB | 50–300ms (warm) | 1 request | Single Adam step, tiny | same rate |

### Song Tower training settings

**Plain English:** Training the Song Brain means showing it thousands of songs and teaching it "these two are similar, those two are different." We use a technique called **InfoNCE** which is like a multiple choice test — given one song, pick its match from a crowd of 512 others. The bigger the crowd (batch size), the harder the test, the better the brain gets at distinguishing songs. This needs a powerful GPU with lots of memory.

| Knob | Value | Rationale |
|------|-------|-----------|
| Optimizer | AdamW | Standard for contrastive; handles weight decay cleanly |
| Learning rate | 3e-4 (warm up 5% steps, cosine decay) | Standard range for MLP contrastive |
| Weight decay | 1e-2 | Regularization |
| Batch size | 512 (4090) / 256 (L4) | InfoNCE quality scales with negatives; 512 fits in 24GB with AMP |
| Epochs | 25–35 | Early-stop on val InfoNCE loss (held-out 10% track IDs) |
| Temperature τ | 0.07 (start), tune in [0.05, 0.15] | Controls sharpness of similarity distribution |
| AMP | `torch.amp.autocast('cuda')` + `GradScaler` | 2× speed, ~40% less VRAM |
| Seed | 42 + cudnn deterministic | Reproducibility |
| Val split | 10% of track IDs (not rows — prevent data leakage) | Validation |
| Checkpoint | Every 5 epochs + best by val loss | Recovery |

### InfoNCE loss (the "multiple choice test")

**Plain English:** Imagine you have 512 songs in a batch. For each song, we create two slightly different "views" of it (by adding tiny random noise). The test is: given view A of song #1, can the model pick out view B of song #1 from the other 511 distractors? If yes → good embedding. If confused → update the brain. The `temperature` controls how "sharp" the model needs to be (lower = pickier).

```python
def info_nce_loss(anchor, positive, temperature=0.07):
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    logits = anchor @ positive.T / temperature  # (B, B) similarity matrix
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)
```

Training creates pairs from the same track with different augmentations (dropout noise: run same input through Song Tower twice with different random masks).

### Serverless endpoint settings (RunPod console checklist)

| Setting | Value | Why |
|---------|-------|-----|
| GPU priority | L4 primary, 4090 secondary | L4 cheaper flex; 4090 as overflow |
| FlashBoot | **On** | Resume from snapshot faster after idle |
| Execution timeout | 600s (600000ms) | Generous for cold starts with model load |
| Idle timeout | 30s (dev) / 60s (demo) | Keep worker warm between judge requests |
| Active workers | 0 (dev) / 1 (judging) | 1 active eliminates cold start; costs idle $/s |
| Max workers | 3 | Cost cap for hackathon |
| Throttled workers | 0 | No throttle queue for demo |
| Container disk | 20 GB | Fits image + artifacts |

### RunPod Serverless API (how Convex calls it)

**Endpoint:** `POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync`

**Request:**
```json
{
  "input": {
    "operation": "rank",
    "sessionTrackIds": ["spotify:track:abc123", "..."],
    "knownTrackIds": ["spotify:track:def456", "..."],
    "context": {
      "hour_sin": 0.5, "hour_cos": -0.87,
      "dow_sin": 0.78, "dow_cos": 0.62,
      "session_skip_rate": 0.3,
      "consecutive_skips_norm": 0.0,
      "session_length_norm": 0.2,
      "volume_delta": 0.0,
      "location_home": 1.0, "location_gym": 0.0,
      "location_transit": 0.0, "location_other": 0.0,
      "weather_temp_norm": 0.5, "weather_is_rain": 0.0,
      "weather_humidity_norm": 0.6, "weather_is_day": 1.0
    },
    "userWeightsBase64": "<base64-encoded state_dict bytes>"
  }
}
```

**Response (success):**
```json
{
  "id": "job-abc123",
  "status": "COMPLETED",
  "output": {
    "uris": ["spotify:track:xyz789", "..."],
    "scores": [0.92, 0.88, "..."],
    "modelBundleId": "v1",
    "inferenceMs": 450
  },
  "delayTime": 12000,
  "executionTime": 1500
}
```

**Convex Workflow (`@convex-dev/workflow`):**

We use Convex Workflows for playlist generation because it's a multi-step process (fetch user weights → call RunPod → save result → trigger online learning). Workflows are **durable** — if the server restarts mid-way, it picks up where it left off. If RunPod is slow or errors, it retries automatically.

```typescript
// convex/workflows.ts
import { WorkflowManager } from "@convex-dev/workflow";
import { components, internal } from "./_generated/api";
import { v } from "convex/values";

export const workflow = new WorkflowManager(components.workflow);

export const generatePlaylist = workflow.define({
  args: {
    userId: v.id("users"),
    sessionTrackIds: v.array(v.string()),
    context: v.object({ /* 16 context fields */ }),
  },
  returns: v.object({
    uris: v.array(v.string()),
    inferenceMs: v.number(),
  }),
  handler: async (step, args) => {
    // Step 1: Fetch user weights + known tracks from DB
    const userData = await step.runQuery(internal.queries.getUserData, {
      userId: args.userId,
    });

    // Step 2: Call RunPod Serverless (with retry on transient failures)
    const result = await step.runAction(
      internal.runpod.callRumsync,
      {
        operation: "rank",
        sessionTrackIds: args.sessionTrackIds,
        knownTrackIds: userData.knownTrackIds,
        context: args.context,
        userWeightsBase64: userData.weightsBase64,
      },
      { retry: { maxAttempts: 3, initialBackoffMs: 1000, base: 2 } },
    );

    // Step 3: Save playlist result
    await step.runMutation(internal.playlists.saveResult, {
      userId: args.userId,
      uris: result.uris,
      modelBundleId: result.modelBundleId,
    });

    return { uris: result.uris, inferenceMs: result.inferenceMs };
  },
});
```

```typescript
// convex/runpod.ts — the actual HTTP call (internal action)
"use node";
import { internalAction } from "./_generated/server";
import { v } from "convex/values";

export const callRunsync = internalAction({
  args: { operation: v.string(), /* ... rest */ },
  handler: async (ctx, args) => {
    const url = `https://api.runpod.ai/v2/${process.env.RUNPOD_ENDPOINT_ID}/runsync?wait=120000`;
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${process.env.RUNPOD_API_KEY}`,
      },
      body: JSON.stringify({ input: args }),
    });
    if (!res.ok) throw new Error(`RunPod ${res.status}: ${await res.text()}`);
    const body = await res.json();
    if (body.status !== "COMPLETED") throw new Error(`RunPod: ${body.status}`);
    return body.output;
  },
});
```

### Worker handler (full implementation outline)

```python
import base64, io, os, time
import numpy as np
import torch
import torch.nn.functional as F
import faiss
import runpod

# --- Load artifacts once at import time (amortize cold start) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FAISS_INDEX = faiss.read_index("/app/artifacts/faiss_song_v1.index")
FAISS_IDS = np.load("/app/artifacts/faiss_song_v1_ids.npy", allow_pickle=True)
SONG_TOWER = torch.jit.load("/app/artifacts/song_tower_v1.pt", map_location=DEVICE)
SONG_TOWER.eval()

CONTEXT_ENABLED = os.environ.get("ATTUNE_CONTEXT_ENABLED", "1") == "1"
GRU_ENABLED = os.environ.get("ATTUNE_GRU_ENABLED", "0") == "1"

if CONTEXT_ENABLED:
    CONTEXT_ENCODER = torch.jit.load("/app/artifacts/context_encoder_v1.pt", map_location=DEVICE)
    CONTEXT_ENCODER.eval()
if GRU_ENABLED:
    GRU_MODEL = torch.jit.load("/app/artifacts/gru_v1.pt", map_location=DEVICE)
    GRU_MODEL.eval()

# Pre-load all song embeddings for dot-product
ALL_EMBEDDINGS = None  # loaded from FAISS reconstruct or separate .npy

def load_user_weights(base64_str):
    buf = base64.b64decode(base64_str)
    state_dict = torch.load(io.BytesIO(buf), map_location=DEVICE, weights_only=True)
    from attune_ml.models.user_tower import UserTower
    model = UserTower()
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model

def handler(event):
    t0 = time.time()
    inp = event["input"]
    op = inp.get("operation", "rank")

    if op == "rank":
        user_tower = load_user_weights(inp["userWeightsBase64"])
        context_vec = torch.tensor([list(inp["context"].values())], dtype=torch.float32, device=DEVICE)

        # User embedding
        with torch.no_grad():
            user_emb = user_tower.dummy_forward(context_vec[:, :17])  # placeholder
            if CONTEXT_ENABLED:
                user_emb = CONTEXT_ENCODER(user_emb, context_vec)
            # FAISS search
            user_np = user_emb.cpu().numpy().astype(np.float32)
            scores, indices = FAISS_INDEX.search(user_np, 500)
            candidate_ids = FAISS_IDS[indices[0]]
            # ... GRU re-rank if enabled ...
            # ... interleave known/new ...
        
        uris = candidate_ids[:20].tolist()
        return {"uris": uris, "modelBundleId": "v1", "inferenceMs": int((time.time()-t0)*1000)}

    elif op == "online_step":
        user_tower = load_user_weights(inp["userWeightsBase64"])
        user_tower.train()
        optimizer = torch.optim.Adam(user_tower.parameters(), lr=1e-5)
        # Build batch from inp["events"] (new + replayed)
        # One step, clip grad 0.3
        torch.nn.utils.clip_grad_norm_(user_tower.parameters(), 0.3)
        optimizer.step()
        # Serialize updated weights
        buf = io.BytesIO()
        torch.save(user_tower.state_dict(), buf)
        new_weights_b64 = base64.b64encode(buf.getvalue()).decode()
        return {"weightsBase64": new_weights_b64, "version": inp.get("version", 0) + 1}

    return {"error": f"unknown operation: {op}"}

runpod.serverless.start({"handler": handler})
```

### Dockerfile (workers/runpod/Dockerfile)

```dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.12 python3-pip && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY workers/runpod/requirements-serverless.txt .
RUN pip install --no-cache-dir -r requirements-serverless.txt

COPY packages/attune_ml/ /app/packages/attune_ml/
RUN pip install -e /app/packages/attune_ml/

COPY ml/export/faiss_song_v1.index /app/artifacts/
COPY ml/export/faiss_song_v1_ids.npy /app/artifacts/
COPY ml/export/song_tower_v1.pt /app/artifacts/
COPY ml/export/context_encoder_v1.pt /app/artifacts/
COPY ml/export/gru_v1.pt /app/artifacts/

COPY workers/runpod/handler.py /app/handler.py

ENV ATTUNE_CONTEXT_ENABLED=1
ENV ATTUNE_GRU_ENABLED=0

CMD ["python3", "handler.py"]
```

### FAISS — the "song map" search engine

**Plain English:** FAISS (Facebook AI Similarity Search) is like a giant map of all ~114k songs. Each song has been placed at a specific coordinate (its 128-number embedding). When we want to find songs "near" the user, we just ask FAISS: "give me the 500 closest points to this location." It does this in milliseconds because it's optimized for exactly this kind of lookup. We use `IndexFlatIP` which means "exact search using dot-product similarity" — no approximation, just brute-force on 114k vectors (small enough to be instant).

### Training artifacts (what each job produces)

| Artifact | File | Size estimate | Produced by | Consumed by | Versioning |
|----------|------|---------------|-------------|-------------|------------|
| Song Tower weights | `ml/export/song_tower_v1.pt` | ~300 KB | Task 5 Pod train | Serverless image COPY | Tag with train date + loss |
| Song Tower config | `ml/export/train_config.yaml` | ~1 KB | Task 5 | Audit / repro | Same tag |
| Song embeddings | `ml/export/embeddings_song_v1.parquet` | ~60 MB | Task 6 forward | FAISS builder | Rebuild when Song Tower changes |
| FAISS index | `ml/export/faiss_song_v1.index` | ~58 MB | Task 6 FAISS | Serverless image | Rebuild with embeddings |
| FAISS ID map | `ml/export/faiss_song_v1_ids.npy` | ~5 MB | Task 6 | Serverless image | Rebuild with embeddings |
| Context Encoder | `ml/export/context_encoder_v1.pt` | ~200 KB | Task 13 Pod | Serverless image | Version per retrain |
| GRU Sequencer | `ml/export/gru_v1.pt` | ~800 KB | Task 14 Pod | Serverless image (feature-flagged) | Version per retrain |
| User weights | Convex `userWeights` table | ~212 KB per user | Cold train + online steps | Passed to Serverless in `input` | Monotonic version counter |

---

## Part D — File structure (complete)

```
attune/
├── pyproject.toml                          # Python workspace: torch, faiss, datasets, pytest, ruff
├── .gitignore                              # ml/export/, .venv/, __pycache__/, data/, node_modules/
├── README.md                               # Project overview + link to this plan
│
├── packages/
│   └── attune_ml/
│       ├── pyproject.toml                  # Package-level deps (or inherit from root)
│       ├── attune_ml/
│       │   ├── __init__.py
│       │   ├── features/
│       │   │   ├── __init__.py
│       │   │   ├── song_v1.py              # features_song_v1(row) → np.ndarray (26,)
│       │   │   ├── user_v1.py              # features_user_v1(history) → np.ndarray (17,)
│       │   │   └── context_v1.py           # features_context_v1(ctx) → np.ndarray (16,)
│       │   ├── models/
│       │   │   ├── __init__.py
│       │   │   ├── song_tower.py           # SongTower(nn.Module)
│       │   │   ├── user_tower.py           # UserTower(nn.Module)
│       │   │   ├── context_encoder.py      # ContextEncoder(nn.Module)
│       │   │   └── session_gru.py          # SessionGRU(nn.Module)
│       │   ├── train/
│       │   │   ├── song_infonce.py         # CLI: python -m attune_ml.train.song_infonce
│       │   │   ├── user_cold.py            # CLI: python -m attune_ml.train.user_cold
│       │   │   ├── train_context.py        # CLI: python -m attune_ml.train.train_context
│       │   │   └── train_gru.py            # CLI: python -m attune_ml.train.train_gru
│       │   ├── index/
│       │   │   ├── export_embeddings.py    # Forward all tracks → parquet
│       │   │   └── build_faiss.py          # Build IndexFlatIP
│       │   └── inference/
│       │       ├── ranker.py               # Full rank pipeline (used by handler)
│       │       └── online_step.py          # Single Adam step logic
│       └── tests/
│           ├── conftest.py
│           ├── fixtures/
│           │   ├── hf_track_row.json       # Single track with all audio features
│           │   └── sample_playback_event.json
│           ├── test_song_v1.py             # Feature vector tests
│           ├── test_user_v1.py
│           ├── test_context_v1.py
│           ├── test_song_tower.py          # Forward shape/norm tests
│           ├── test_user_tower.py
│           ├── test_context_encoder.py
│           ├── test_gru.py
│           ├── test_song_infonce_shapes.py # Training step smoke
│           ├── test_ranker.py              # Integration: full rank pipeline
│           └── test_schemas.py             # JSON Schema validation
│
├── convex/
│   ├── convex.config.ts                    # App config: registers @convex-dev/workflow component
│   ├── schema.ts                           # Table definitions
│   ├── spotify.ts                          # link mutation (code exchange)
│   ├── events.ts                           # appendBatch mutation
│   ├── workflows.ts                        # WorkflowManager + generatePlaylist workflow
│   ├── runpod.ts                           # internal actions: callRunsync, callOnlineStep
│   ├── learning.ts                         # onlineStep workflow (→ RunPod)
│   ├── onboarding.ts                       # onboarding workflow (fetch + cold train)
│   ├── queries.ts                          # getUserData, getUserWeights queries
│   ├── weather.ts                          # getWeather action (→ OpenWeather API)
│   └── tsconfig.json
│
├── workers/
│   └── runpod/
│       ├── handler.py                      # runpod.serverless.start handler
│       ├── Dockerfile
│       ├── requirements-serverless.txt     # Pinned: torch, faiss-gpu, numpy, runpod
│       └── README.md                       # Deploy instructions
│
├── app/                                    # Expo (React Native + Web)
│   ├── app.json                            # Expo config
│   ├── package.json
│   ├── app/
│   │   ├── (tabs)/
│   │   │   ├── index.tsx                   # Main "Attune this moment" screen
│   │   │   ├── now-playing.tsx             # Now playing + controls
│   │   │   └── settings.tsx                # Unlink, privacy, location override
│   │   ├── login.tsx                       # Spotify PKCE flow
│   │   └── onboarding.tsx                  # Progress UI during cold train
│   ├── components/
│   │   ├── NowPlayingStrip.tsx
│   │   ├── AttuneButton.tsx
│   │   └── ProgressRing.tsx
│   ├── hooks/
│   │   ├── useSpotifyAuth.ts              # PKCE + code exchange
│   │   ├── useTelemetry.ts                # Event batching + context collection
│   │   ├── usePlayback.ts                 # Native Remote / Web SDK abstraction
│   │   └── useWeather.ts                  # Fetch weather from Convex action
│   ├── lib/
│   │   ├── convex.ts                       # ConvexProvider setup
│   │   ├── spotify.ts                      # SDK helpers
│   │   └── context.ts                      # Build 16-d context from device state
│   └── convex/                             # Generated Convex client types
│
├── datasets/
│   ├── schemas/
│   │   ├── playback_event_v1.json
│   │   ├── playlist_request_v1.json
│   │   └── playlist_response_v1.json
│   ├── scripts/
│   │   └── materialize_hf.py              # HF → data/hf_train.parquet
│   └── LOCK                                # Dataset revision hash
│
├── ml/
│   ├── export/                             # .gitignored: checkpoints, FAISS
│   └── evaluation/
│       └── benchmarks.md                   # GPU, wall-clock, loss, latency logs
│
└── docs/
    └── superpowers/
        └── plans/
            └── 2026-05-09-attune-implementation-plan.md  # This file
```

---

## Part E — Tasks (detailed)

### Task 0: Repo bootstrap

**Creates:** `pyproject.toml`, `.gitignore`

- [ ] **Step 1:** Write `pyproject.toml`:

```toml
[project]
name = "attune-workspace"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.3",
    "numpy>=1.26",
    "datasets>=2.19",
    "pandas>=2.2",
    "pytest>=8.0",
    "ruff>=0.4",
    "httpx>=0.27",
    "jsonschema>=4.22",
    "faiss-cpu>=1.8",
]

[project.optional-dependencies]
gpu = ["faiss-gpu>=1.8"]
serverless = ["runpod>=1.6"]

[tool.pytest.ini_options]
testpaths = ["packages/attune_ml/tests"]
```

- [ ] **Step 2:** Write `.gitignore`:

```
ml/export/
data/
.venv/
__pycache__/
*.pyc
node_modules/
.expo/
dist/
*.egg-info/
```

- [ ] **Step 3:** Create directories: `packages/attune_ml/attune_ml/`, `packages/attune_ml/tests/`, `convex/`, `workers/runpod/`, `app/`, `datasets/schemas/`, `datasets/scripts/`, `ml/export/`, `ml/evaluation/`, `docs/`

- [ ] **Step 4:** Commit:

```bash
git add pyproject.toml .gitignore
git commit -m "chore: bootstrap repo structure"
```

---

### Task 1: JSON Schemas (contracts)

**Creates:** `datasets/schemas/playback_event_v1.json`, `playlist_request_v1.json`, `playlist_response_v1.json`, test fixture, test file

- [ ] **Step 1:** Write `datasets/schemas/playback_event_v1.json`:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "PlaybackEvent v1",
  "type": "object",
  "required": ["event_id", "user_id", "spotify_track_uri", "listen_ratio", "skip", "client_schema_version"],
  "properties": {
    "event_id": { "type": "string", "format": "uuid" },
    "user_id": { "type": "string" },
    "spotify_track_uri": { "type": "string", "pattern": "^spotify:track:" },
    "listen_ratio": { "type": "number", "minimum": 0, "maximum": 1 },
    "skip": { "type": "boolean" },
    "skip_bucket": { "type": "string", "enum": ["instant", "early", "late", "none"] },
    "skip_position_ms": { "type": "integer", "minimum": 0 },
    "replay": { "type": "boolean" },
    "position_ms": { "type": "integer", "minimum": 0 },
    "duration_ms": { "type": "integer", "minimum": 0 },
    "volume_delta": { "type": "number", "minimum": -1, "maximum": 1 },
    "hour_local": { "type": "integer", "minimum": 0, "maximum": 23 },
    "dow": { "type": "integer", "minimum": 0, "maximum": 6 },
    "location_bucket": { "type": "string", "enum": ["home", "gym", "transit", "other"] },
    "session_skip_rate": { "type": "number", "minimum": 0, "maximum": 1 },
    "consecutive_skips": { "type": "integer", "minimum": 0 },
    "weather": {
      "type": "object",
      "properties": {
        "temp": { "type": "number" },
        "condition": { "type": "string", "enum": ["clear", "clouds", "rain", "snow", "storm"] },
        "humidity": { "type": "integer", "minimum": 0, "maximum": 100 },
        "is_day": { "type": "boolean" }
      }
    },
    "client_schema_version": { "type": "string", "const": "1" },
    "timestamp_ms": { "type": "integer" }
  }
}
```

- [ ] **Step 2:** Write fixture `packages/attune_ml/tests/fixtures/sample_playback_event.json`:

```json
{
  "event_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user_abc123",
  "spotify_track_uri": "spotify:track:4iV5W9uYEdYUVa79Axb7Rh",
  "listen_ratio": 0.85,
  "skip": false,
  "skip_bucket": "none",
  "position_ms": 180000,
  "duration_ms": 210000,
  "volume_delta": 0.0,
  "hour_local": 14,
  "dow": 2,
  "location_bucket": "home",
  "session_skip_rate": 0.1,
  "consecutive_skips": 0,
  "weather": { "temp": 22.5, "condition": "clear", "humidity": 45, "is_day": true },
  "client_schema_version": "1",
  "timestamp_ms": 1715270400000
}
```

- [ ] **Step 3:** Write test `packages/attune_ml/tests/test_schemas.py`:

```python
import json
from pathlib import Path
import jsonschema
import pytest

SCHEMA_DIR = Path("datasets/schemas")
FIXTURE_DIR = Path("packages/attune_ml/tests/fixtures")

def test_playback_event_validates_against_schema():
    schema = json.loads((SCHEMA_DIR / "playback_event_v1.json").read_text())
    sample = json.loads((FIXTURE_DIR / "sample_playback_event.json").read_text())
    jsonschema.validate(instance=sample, schema=schema)

def test_playback_event_rejects_invalid():
    schema = json.loads((SCHEMA_DIR / "playback_event_v1.json").read_text())
    invalid = {"event_id": "not-a-valid-event"}  # missing required fields
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid, schema=schema)
```

- [ ] **Step 4:** Run tests — expect PASS:

```bash
cd /Users/leo/personal/winners-nia && python -m pytest packages/attune_ml/tests/test_schemas.py -v
```

- [ ] **Step 5:** Commit:

```bash
git add datasets/schemas packages/attune_ml/tests
git commit -m "feat: add v1 JSON schemas and validation tests"
```

---

### Task 2: Song features (26-d)

**Creates:** `packages/attune_ml/attune_ml/features/__init__.py`, `song_v1.py`, test fixtures, test file

- [ ] **Step 1:** Write fixture `packages/attune_ml/tests/fixtures/hf_track_row.json`:

```json
{
  "track_id": "5SuOikwiRyPMVoIQDJUgSV",
  "track_name": "Test Track",
  "key": 1,
  "danceability": 0.5,
  "energy": 0.5,
  "speechiness": 0.1,
  "acousticness": 0.2,
  "instrumentalness": 0.0,
  "liveness": 0.1,
  "valence": 0.6,
  "loudness": -10.0,
  "tempo": 120.0,
  "mode": 1,
  "explicit": false,
  "popularity": 50,
  "duration_ms": 200000,
  "time_signature": 4
}
```

- [ ] **Step 2:** Write failing test `packages/attune_ml/tests/test_song_v1.py`:

```python
import json
from pathlib import Path
import numpy as np
import pytest
from attune_ml.features.song_v1 import features_song_v1

FIXTURE = Path("packages/attune_ml/tests/fixtures/hf_track_row.json")

def test_shape_and_dtype():
    row = json.loads(FIXTURE.read_text())
    v = features_song_v1(row)
    assert v.shape == (26,)
    assert v.dtype == np.float32
    assert np.isfinite(v).all()

def test_key_one_hot():
    row = json.loads(FIXTURE.read_text())
    v = features_song_v1(row)
    assert v[0] == 0.0  # key=1 → index 1 is hot
    assert v[1] == 1.0
    assert v[2] == 0.0

def test_key_missing():
    row = {"key": -1, "danceability": 0.5, "energy": 0.5, "speechiness": 0.1,
           "acousticness": 0.2, "instrumentalness": 0.0, "liveness": 0.1,
           "valence": 0.6, "loudness": -10.0, "tempo": 120.0, "mode": 1,
           "explicit": False, "popularity": 50, "duration_ms": 200000, "time_signature": 4}
    v = features_song_v1(row)
    assert v[:12].sum() == 0.0  # all zeros

def test_scalar_values():
    row = json.loads(FIXTURE.read_text())
    v = features_song_v1(row)
    assert v[12] == pytest.approx(0.5)   # danceability
    assert v[13] == pytest.approx(0.5)   # energy
    assert v[19] == pytest.approx((-10.0 + 60.0) / 60.0)  # norm_loudness = 0.833...
    assert v[20] == pytest.approx(120.0 / 240.0)           # tempo_norm = 0.5
    assert v[21] == pytest.approx(1.0)   # mode
    assert v[22] == pytest.approx(0.0)   # explicit=false
    assert v[23] == pytest.approx(50.0 / 100.0)            # popularity_norm
    assert v[24] == pytest.approx(200_000 / 330_000)       # duration_norm
    assert v[25] == pytest.approx(4.0 / 7.0)              # time_signature_norm

def test_all_values_in_range():
    row = json.loads(FIXTURE.read_text())
    v = features_song_v1(row)
    assert (v >= 0.0).all()
    assert (v <= 1.01).all()  # slight tolerance for tempo > 240
```

- [ ] **Step 3:** Run — expect FAIL (module not found):

```bash
python -m pytest packages/attune_ml/tests/test_song_v1.py -v
```

- [ ] **Step 4:** Implement `packages/attune_ml/attune_ml/features/song_v1.py`:

```python
import numpy as np

def features_song_v1(row: dict) -> np.ndarray:
    key = int(row.get("key", -1))
    oh = np.zeros(12, dtype=np.float32)
    if 0 <= key <= 11:
        oh[key] = 1.0

    loud = float(row.get("loudness", -60))
    norm_loud = max(0.0, min(1.0, (loud + 60.0) / 60.0))

    scalars = np.array([
        float(row["danceability"]),
        float(row["energy"]),
        float(row["speechiness"]),
        float(row["acousticness"]),
        float(row["instrumentalness"]),
        float(row["liveness"]),
        float(row["valence"]),
        norm_loud,
        float(row.get("tempo", 0)) / 240.0,
        float(row.get("mode", 0)),
        1.0 if row.get("explicit") in (True, 1, "true") else 0.0,
        float(row.get("popularity", 0)) / 100.0,
        min(1.0, float(row.get("duration_ms", 0)) / 330_000.0),
        min(1.0, max(0.0, float(row.get("time_signature", 4)) / 7.0)),
    ], dtype=np.float32)

    return np.concatenate([oh, scalars], axis=0)
```

- [ ] **Step 5:** Create `__init__.py` files.

- [ ] **Step 6:** Run — expect PASS:

```bash
python -m pytest packages/attune_ml/tests/test_song_v1.py -v
```

- [ ] **Step 7:** Commit:

```bash
git add packages/attune_ml
git commit -m "feat: add features_song_v1 (26-d vector)"
```

---

### Task 3: Song Tower module

**Creates:** `packages/attune_ml/attune_ml/models/song_tower.py`, `__init__.py`, test

- [ ] **Step 1:** Write test `packages/attune_ml/tests/test_song_tower.py`:

```python
import torch
import pytest
from attune_ml.models.song_tower import SongTower

def test_forward_shape():
    model = SongTower()
    x = torch.randn(4, 26)
    out = model(x)
    assert out.shape == (4, 128)

def test_output_l2_normalized():
    model = SongTower()
    x = torch.randn(8, 26)
    out = model(x)
    norms = torch.norm(out, dim=-1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5)

def test_deterministic_with_eval():
    model = SongTower()
    model.eval()
    x = torch.randn(2, 26)
    out1 = model(x)
    out2 = model(x)
    assert torch.allclose(out1, out2)
```

- [ ] **Step 2:** Run — expect FAIL.

- [ ] **Step 3:** Implement `packages/attune_ml/attune_ml/models/song_tower.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SongTower(nn.Module):
    def __init__(self, input_dim: int = 26, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x, dim=-1)
```

- [ ] **Step 4:** Run — expect PASS.

- [ ] **Step 5:** Commit `feat: SongTower MLP`

---

### Task 4: HF data materialization

**Creates:** `datasets/scripts/materialize_hf.py`, `datasets/LOCK`

- [ ] **Step 1:** Write `datasets/scripts/materialize_hf.py`:

```python
"""Download maharshipandya/spotify-tracks-dataset and save as parquet."""
import os
from pathlib import Path
from datasets import load_dataset

def main():
    ds = load_dataset("maharshipandya/spotify-tracks-dataset", split="train")
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "hf_train.parquet"
    ds.to_parquet(str(out_path))
    print(f"Saved {len(ds)} rows to {out_path}")
    
    # Write lock file with dataset fingerprint
    lock_path = Path("datasets/LOCK")
    lock_path.write_text(f"dataset: maharshipandya/spotify-tracks-dataset\nrows: {len(ds)}\nfingerprint: {ds._fingerprint}\n")
    print(f"Lock: {lock_path}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2:** Run (requires network):

```bash
python datasets/scripts/materialize_hf.py
```

Expected output: `Saved 114000 rows to data/hf_train.parquet` (exact count may vary).

- [ ] **Step 3:** Verify `data/` is gitignored, commit script + lock:

```bash
git add datasets/scripts/materialize_hf.py datasets/LOCK
git commit -m "feat: HF materialize script"
```

---

### Task 5: Song Tower training CLI (**RTX 4090 Pod**)

**Creates:** `packages/attune_ml/attune_ml/train/song_infonce.py`, train shape test, `ml/evaluation/benchmarks.md`

**Hardware:** Start a RunPod Pod with **RTX 4090 24GB**. Clone repo, `pip install -e packages/attune_ml[gpu]`.

- [ ] **Step 1:** Write shape test `packages/attune_ml/tests/test_song_infonce_shapes.py`:

```python
import torch
from attune_ml.models.song_tower import SongTower
from attune_ml.train.song_infonce import info_nce_loss, make_training_pair

def test_info_nce_loss_finite():
    model = SongTower()
    batch = torch.randn(8, 26)
    anchor = model(batch)
    positive = model(batch)  # Same input, different dropout
    loss = info_nce_loss(anchor, positive, temperature=0.07)
    assert loss.isfinite()
    assert loss.item() > 0

def test_gradients_flow():
    model = SongTower()
    batch = torch.randn(8, 26)
    anchor = model(batch)
    positive = model(batch)
    loss = info_nce_loss(anchor, positive, temperature=0.07)
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None
        assert p.grad.abs().sum() > 0
```

- [ ] **Step 2:** Implement `packages/attune_ml/attune_ml/train/song_infonce.py`:

```python
"""Song Tower InfoNCE training CLI.

Usage:
    python -m attune_ml.train.song_infonce \
        --data data/hf_train.parquet \
        --epochs 30 --batch 512 --lr 3e-4 --tau 0.07 \
        --output ml/export/song_tower_v1.pt \
        [--limit-rows 8192]  # for smoke testing
"""
import argparse, time, yaml
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
from attune_ml.models.song_tower import SongTower
from attune_ml.features.song_v1 import features_song_v1

def info_nce_loss(anchor, positive, temperature=0.07):
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    logits = anchor @ positive.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)

def make_training_pair(features_batch, dropout_rate=0.1):
    """Create positive pair by applying independent dropout noise."""
    mask1 = torch.bernoulli(torch.full_like(features_batch, 1 - dropout_rate))
    mask2 = torch.bernoulli(torch.full_like(features_batch, 1 - dropout_rate))
    return features_batch * mask1, features_batch * mask2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/hf_train.parquet")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--output", type=str, default="ml/export/song_tower_v1.pt")
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load and featurize
    df = pd.read_parquet(args.data)
    if args.limit_rows:
        df = df.head(args.limit_rows)
    
    features = np.stack([features_song_v1(row) for _, row in df.iterrows()])
    tensor = torch.from_numpy(features).to(device)
    loader = DataLoader(TensorDataset(tensor), batch_size=args.batch, shuffle=True, drop_last=True)

    model = SongTower().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()

    best_loss = float("inf")
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for (batch,) in loader:
            anchor, positive = make_training_pair(batch)
            optimizer.zero_grad()
            with autocast():
                emb_a = model(anchor)
                emb_p = model(positive)
                loss = info_nce_loss(emb_a, emb_p, args.tau)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs}  loss={avg_loss:.4f}  elapsed={elapsed:.0f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output)

    # Save config
    config = vars(args) | {"best_loss": best_loss, "wall_seconds": time.time() - t0, "device": device}
    config_path = Path(args.output).with_suffix(".yaml")
    config_path.write_text(yaml.dump(config))
    print(f"Done. Best loss: {best_loss:.4f}. Saved to {args.output}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3:** Run tests locally:

```bash
python -m pytest packages/attune_ml/tests/test_song_infonce_shapes.py -v
```

- [ ] **Step 4:** Smoke on Pod (should complete <3 min):

```bash
python -m attune_ml.train.song_infonce --epochs 1 --batch 512 --limit-rows 8192 --output ml/export/song_tower_smoke.pt
```

- [ ] **Step 5:** Full train on Pod:

```bash
python -m attune_ml.train.song_infonce --epochs 30 --batch 512 --lr 3e-4 --tau 0.07 --output ml/export/song_tower_v1.pt
```

Record in `ml/evaluation/benchmarks.md`:
```markdown
## Song Tower v1
- GPU: RTX 4090 24GB
- Batch: 512, Epochs: 30, Tau: 0.07
- Final loss: X.XXXX
- Wall time: XX min
- Date: YYYY-MM-DD
```

- [ ] **Step 6:** Also export as TorchScript for Serverless:

```python
model = SongTower()
model.load_state_dict(torch.load("ml/export/song_tower_v1.pt"))
model.eval()
scripted = torch.jit.script(model)
scripted.save("ml/export/song_tower_v1_jit.pt")
```

- [ ] **Step 7:** Commit:

```bash
git add packages/attune_ml/attune_ml/train/song_infonce.py packages/attune_ml/tests/test_song_infonce_shapes.py ml/evaluation/benchmarks.md
git commit -m "feat: song tower InfoNCE training loop"
```

---

### Task 6: Export embeddings + build FAISS (**CPU**)

**Creates:** `packages/attune_ml/attune_ml/index/export_embeddings.py`, `build_faiss.py`

**Hardware:** CPU Pod or laptop. GPU optional for faster forward (not required for 114k).

- [ ] **Step 1:** Implement `export_embeddings.py`:

```python
"""Forward all tracks through Song Tower → embeddings parquet."""
import argparse
import numpy as np
import pandas as pd
import torch
from attune_ml.models.song_tower import SongTower
from attune_ml.features.song_v1 import features_song_v1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/hf_train.parquet")
    parser.add_argument("--checkpoint", default="ml/export/song_tower_v1.pt")
    parser.add_argument("--output", default="ml/export/embeddings_song_v1.parquet")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SongTower()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    df = pd.read_parquet(args.data)
    features = np.stack([features_song_v1(row) for _, row in df.iterrows()])

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(features), args.batch_size):
            batch = torch.from_numpy(features[i:i+args.batch_size]).to(device)
            emb = model(batch).cpu().numpy()
            embeddings.append(emb)

    all_emb = np.concatenate(embeddings, axis=0)
    track_ids = df["track_id"].values if "track_id" in df.columns else df.index.astype(str).values

    out_df = pd.DataFrame({"track_id": track_ids})
    for i in range(128):
        out_df[f"emb_{i}"] = all_emb[:, i]
    out_df.to_parquet(args.output, index=False)
    print(f"Exported {len(out_df)} embeddings to {args.output}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2:** Implement `build_faiss.py`:

```python
"""Build FAISS IndexFlatIP from embeddings parquet."""
import argparse
import numpy as np
import pandas as pd
import faiss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="ml/export/embeddings_song_v1.parquet")
    parser.add_argument("--index-output", default="ml/export/faiss_song_v1.index")
    parser.add_argument("--ids-output", default="ml/export/faiss_song_v1_ids.npy")
    args = parser.parse_args()

    df = pd.read_parquet(args.embeddings)
    track_ids = df["track_id"].values
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    vectors = df[emb_cols].values.astype(np.float32)

    # L2-normalize (should already be normalized, but ensure)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-8)

    index = faiss.IndexFlatIP(128)
    index.add(vectors)

    faiss.write_index(index, args.index_output)
    np.save(args.ids_output, track_ids)
    print(f"Index: {index.ntotal} vectors, saved to {args.index_output} ({os.path.getsize(args.index_output)/1e6:.1f} MB)")

if __name__ == "__main__":
    import os
    main()
```

- [ ] **Step 3:** Run:

```bash
python -m attune_ml.index.export_embeddings
python -m attune_ml.index.build_faiss
```

- [ ] **Step 4:** Smoke test:

```python
import faiss, numpy as np
index = faiss.read_index("ml/export/faiss_song_v1.index")
ids = np.load("ml/export/faiss_song_v1_ids.npy", allow_pickle=True)
query = np.random.randn(1, 128).astype(np.float32)
query /= np.linalg.norm(query)
scores, indices = index.search(query, 10)
print("Top 10 IDs:", ids[indices[0]])
print("Scores:", scores[0])
```

- [ ] **Step 5:** Record in benchmarks: index size, ntotal, build time.

- [ ] **Step 6:** Commit:

```bash
git add packages/attune_ml/attune_ml/index/
git commit -m "feat: FAISS index builder + embedding export"
```

---

### Task 7: Convex project + schema + workflow component

**Creates:** `convex/schema.ts`, `convex/convex.config.ts`, `convex/tsconfig.json`, `package.json`

- [ ] **Step 1:** Initialize Convex + install workflow component:

```bash
cd /Users/leo/personal/winners-nia
npm init -y  # or ensure package.json exists
npm install convex @convex-dev/workflow
npx convex init
```

- [ ] **Step 1b:** Write `convex/convex.config.ts`:

```typescript
import { defineApp } from "convex/server";
import workflow from "@convex-dev/workflow/convex.config";

export default defineApp({
  components: {
    workflow,
  },
});
```

- [ ] **Step 2:** Write `convex/schema.ts`:

```typescript
import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

export default defineSchema({
  users: defineTable({
    spotifyId: v.string(),
    encryptedRefresh: v.string(),
    displayName: v.optional(v.string()),
    onboardingStatus: v.string(), // "pending" | "fetching" | "training" | "ready"
    createdAt: v.number(),
  }).index("by_spotifyId", ["spotifyId"]),

  playbackEvents: defineTable({
    userId: v.id("users"),
    eventId: v.string(),
    spotifyTrackUri: v.string(),
    listenRatio: v.number(),
    skip: v.boolean(),
    skipBucket: v.optional(v.string()),
    replay: v.optional(v.boolean()),
    positionMs: v.optional(v.number()),
    durationMs: v.optional(v.number()),
    volumeDelta: v.optional(v.number()),
    hourLocal: v.optional(v.number()),
    dow: v.optional(v.number()),
    locationBucket: v.optional(v.string()),
    sessionSkipRate: v.optional(v.number()),
    consecutiveSkips: v.optional(v.number()),
    weather: v.optional(v.object({
      temp: v.number(),
      condition: v.string(),
      humidity: v.number(),
      isDay: v.boolean(),
    })),
    idempotencyKey: v.string(),
    timestampMs: v.number(),
  })
    .index("by_userId_time", ["userId", "timestampMs"])
    .index("by_userId_idempotency", ["userId", "idempotencyKey"]),

  userWeights: defineTable({
    userId: v.id("users"),
    weightsBase64: v.string(),
    version: v.number(),
    updatedAt: v.number(),
  }).index("by_userId", ["userId"]),

  onboardingJobs: defineTable({
    userId: v.id("users"),
    status: v.string(), // "pending" | "fetching" | "training" | "ready" | "error"
    error: v.optional(v.string()),
    startedAt: v.number(),
    completedAt: v.optional(v.number()),
  }).index("by_userId", ["userId"]),

  trackCache: defineTable({
    spotifyTrackUri: v.string(),
    audioFeatures: v.object({
      danceability: v.number(),
      energy: v.number(),
      speechiness: v.number(),
      acousticness: v.number(),
      instrumentalness: v.number(),
      liveness: v.number(),
      valence: v.number(),
      loudness: v.number(),
      tempo: v.number(),
      key: v.number(),
      mode: v.number(),
      timeSignature: v.number(),
      durationMs: v.number(),
    }),
  }).index("by_uri", ["spotifyTrackUri"]),
});
```

- [ ] **Step 3:** Deploy:

```bash
npx convex dev
```

Expect: schema validation passes, tables created.

- [ ] **Step 4:** Commit:

```bash
git add convex/ package.json package-lock.json
git commit -m "feat: convex schema v1"
```

---

### Task 8: Event ingestion

**Creates:** `convex/events.ts`

- [ ] **Step 1:** Write `convex/events.ts`:

```typescript
import { mutation } from "./_generated/server";
import { v } from "convex/values";

export const appendBatch = mutation({
  args: {
    userId: v.id("users"),
    idempotencyKey: v.string(),
    events: v.array(v.object({
      eventId: v.string(),
      spotifyTrackUri: v.string(),
      listenRatio: v.number(),
      skip: v.boolean(),
      skipBucket: v.optional(v.string()),
      replay: v.optional(v.boolean()),
      positionMs: v.optional(v.number()),
      durationMs: v.optional(v.number()),
      volumeDelta: v.optional(v.number()),
      hourLocal: v.optional(v.number()),
      dow: v.optional(v.number()),
      locationBucket: v.optional(v.string()),
      sessionSkipRate: v.optional(v.number()),
      consecutiveSkips: v.optional(v.number()),
      weather: v.optional(v.object({
        temp: v.number(),
        condition: v.string(),
        humidity: v.number(),
        isDay: v.boolean(),
      })),
      timestampMs: v.number(),
    })),
  },
  handler: async (ctx, args) => {
    // Idempotency check
    const existing = await ctx.db
      .query("playbackEvents")
      .withIndex("by_userId_idempotency", (q) =>
        q.eq("userId", args.userId).eq("idempotencyKey", args.idempotencyKey)
      )
      .first();

    if (existing) return { inserted: 0, duplicate: true };

    let inserted = 0;
    for (const event of args.events) {
      await ctx.db.insert("playbackEvents", {
        userId: args.userId,
        idempotencyKey: args.idempotencyKey,
        ...event,
      });
      inserted++;
    }
    return { inserted, duplicate: false };
  },
});
```

- [ ] **Step 2:** Test in Convex dashboard: call `appendBatch` twice with same idempotencyKey → second returns `{ inserted: 0, duplicate: true }`.

- [ ] **Step 3:** Commit:

```bash
git add convex/events.ts
git commit -m "feat: event ingest with idempotency"
```

---

### Task 9: Spotify link + cold User Tower

**Creates:** `convex/spotify.ts`, `convex/onboarding.ts`, `packages/attune_ml/attune_ml/train/user_cold.py`, `packages/attune_ml/attune_ml/features/user_v1.py`

- [ ] **Step 1:** Write `convex/spotify.ts`:

```typescript
"use node";
import { action } from "./_generated/server";
import { v } from "convex/values";
import { internal } from "./_generated/api";

export const link = action({
  args: {
    code: v.string(),
    redirectUri: v.string(),
  },
  handler: async (ctx, args) => {
    const clientId = process.env.SPOTIFY_CLIENT_ID!;
    const clientSecret = process.env.SPOTIFY_CLIENT_SECRET!;

    // Exchange code for tokens
    const tokenRes = await fetch("https://accounts.spotify.com/api/token", {
      method: "POST",
      headers: { "content-type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        grant_type: "authorization_code",
        code: args.code,
        redirect_uri: args.redirectUri,
        client_id: clientId,
        client_secret: clientSecret,
      }),
    });

    if (!tokenRes.ok) throw new Error(`Spotify token exchange failed: ${await tokenRes.text()}`);
    const tokens = await tokenRes.json();

    // Get Spotify user profile
    const profileRes = await fetch("https://api.spotify.com/v1/me", {
      headers: { authorization: `Bearer ${tokens.access_token}` },
    });
    const profile = await profileRes.json();

    // Upsert user with refresh token
    const userId = await ctx.runMutation(internal.spotify.upsertUser, {
      spotifyId: profile.id,
      displayName: profile.display_name,
      encryptedRefresh: tokens.refresh_token, // TODO: encrypt in production
    });

    // Start onboarding
    await ctx.runMutation(internal.onboarding.createJob, { userId });

    // Trigger background data fetch + cold train
    await ctx.scheduler.runAfter(0, internal.onboarding.fetchAndTrain, { userId });

    return { userId, spotifyId: profile.id };
  },
});
```

- [ ] **Step 2:** Write `convex/onboarding.ts` (internal actions for fetching Spotify data and triggering cold train).

- [ ] **Step 3:** Implement `packages/attune_ml/attune_ml/features/user_v1.py`:

```python
import numpy as np

def features_user_v1(track_features: list[np.ndarray], history_stats: dict) -> np.ndarray:
    """Compute 17-d user feature vector from their track history."""
    if not track_features:
        return np.zeros(17, dtype=np.float32)

    stacked = np.stack(track_features)  # (N, 26) song features

    return np.array([
        stacked[:, 12].mean(),  # mean_danceability
        stacked[:, 13].mean(),  # mean_energy
        stacked[:, 18].mean(),  # mean_valence
        stacked[:, 15].mean(),  # mean_acousticness
        stacked[:, 16].mean(),  # mean_instrumentalness
        stacked[:, 20].mean(),  # mean_tempo_norm
        stacked[:, 19].mean(),  # mean_loudness_norm
        history_stats.get("genre_diversity", 0.5),
        history_stats.get("listening_recency", 0.5),
        history_stats.get("skip_rate_30d", 0.5),
        history_stats.get("replay_rate_30d", 0.1),
        history_stats.get("avg_listen_ratio", 0.7),
        history_stats.get("discovery_ratio", 0.3),
        history_stats.get("session_length_avg", 0.5),
        np.sin(2 * np.pi * history_stats.get("peak_hour", 12) / 24),
        np.cos(2 * np.pi * history_stats.get("peak_hour", 12) / 24),
        min(1.0, history_stats.get("account_age_days", 0) / 365.0),
    ], dtype=np.float32)
```

- [ ] **Step 4:** Implement `packages/attune_ml/attune_ml/train/user_cold.py`:

```python
"""Cold User Tower training from Spotify history.

Usage:
    python -m attune_ml.train.user_cold \
        --history history.json \
        --song-tower ml/export/song_tower_v1.pt \
        --output user_tower_init.pt \
        --epochs 100
"""
import argparse, json, time, io, base64
import numpy as np
import torch
import torch.nn.functional as F
from attune_ml.models.user_tower import UserTower
from attune_ml.models.song_tower import SongTower
from attune_ml.features.song_v1 import features_song_v1
from attune_ml.features.user_v1 import features_user_v1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=str, required=True)
    parser.add_argument("--song-tower", type=str, default="ml/export/song_tower_v1.pt")
    parser.add_argument("--output", type=str, default="ml/export/user_tower_init.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = "cpu"  # Cold train on CPU is fine
    t0 = time.time()

    # Load Song Tower (frozen)
    song_tower = SongTower()
    song_tower.load_state_dict(torch.load(args.song_tower, map_location=device))
    song_tower.eval()

    # Load user history
    history = json.loads(open(args.history).read())
    positive_tracks = history["positives"]  # tracks user liked/saved/completed
    negative_tracks = history["negatives"]  # random non-interacted

    # Featurize and embed
    pos_features = [features_song_v1(t) for t in positive_tracks]
    neg_features = [features_song_v1(t) for t in negative_tracks]

    with torch.no_grad():
        pos_embs = song_tower(torch.from_numpy(np.stack(pos_features)))
        neg_embs = song_tower(torch.from_numpy(np.stack(neg_features)))

    # User features
    user_feat = features_user_v1(pos_features, history.get("stats", {}))
    user_input = torch.from_numpy(user_feat).unsqueeze(0)  # (1, 17)

    # Train User Tower to produce embedding close to positive song embeddings
    user_tower = UserTower()
    user_tower.to(device)
    optimizer = torch.optim.Adam(user_tower.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        user_emb = user_tower(user_input)  # (1, 128)
        pos_sim = (user_emb @ pos_embs.T).squeeze(0)  # (N_pos,)
        neg_sim = (user_emb @ neg_embs.T).squeeze(0)  # (N_neg,)

        # Hinge-style loss: want pos_sim > neg_sim by margin
        margin = 0.2
        loss = F.relu(margin - pos_sim.mean() + neg_sim.mean())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    wall = time.time() - t0
    torch.save(user_tower.state_dict(), args.output)
    print(f"Cold User train done in {wall:.1f}s. Saved to {args.output}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 5:** Implement `packages/attune_ml/attune_ml/models/user_tower.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    def __init__(self, input_dim: int = 17, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x, dim=-1)
```

- [ ] **Step 6:** Test cold train with mock data.

- [ ] **Step 7:** Commit:

```bash
git add convex/spotify.ts convex/onboarding.ts packages/attune_ml/attune_ml/features/user_v1.py packages/attune_ml/attune_ml/train/user_cold.py packages/attune_ml/attune_ml/models/user_tower.py
git commit -m "feat: spotify link + cold user tower training"
```

---

### Task 10: RunPod Serverless worker (**L4/4090 PRO endpoint**)

**Creates:** `workers/runpod/handler.py`, `Dockerfile`, `requirements-serverless.txt`, `README.md`

- [ ] **Step 1:** Write `workers/runpod/requirements-serverless.txt`:

```
torch==2.3.1+cu124
faiss-gpu==1.8.0
numpy>=1.26
runpod>=1.6
```

- [ ] **Step 2:** Write `workers/runpod/handler.py` (see full implementation in Part C above).

- [ ] **Step 3:** Write `workers/runpod/Dockerfile` (see Part C above).

- [ ] **Step 4:** Local test:

```bash
cd workers/runpod
pip install runpod
python handler.py --rp_serve_api  # starts local test server
curl -X POST http://localhost:8000/runsync \
  -H "content-type: application/json" \
  -d '{"input": {"operation": "rank", "userWeightsBase64": "...", "context": {...}, "sessionTrackIds": [], "knownTrackIds": []}}'
```

Expect: response with `output.uris` array, latency <500ms after first load.

- [ ] **Step 5:** Build and push Docker image:

```bash
docker build -t attune-worker:v1 -f workers/runpod/Dockerfile .
docker tag attune-worker:v1 <registry>/attune-worker:v1
docker push <registry>/attune-worker:v1
```

- [ ] **Step 6:** RunPod console:
  1. Create new **Serverless Endpoint**
  2. Set Docker image: `<registry>/attune-worker:v1`
  3. GPU: L4 primary, 4090 secondary
  4. FlashBoot: on
  5. Execution timeout: 600s
  6. Active workers: 0 (set to 1 for demo)
  7. Max workers: 3
  8. Copy **Endpoint ID**

- [ ] **Step 7:** Test live endpoint:

```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "content-type: application/json" \
  -H "authorization: Bearer <API_KEY>" \
  -d '{"input": {"operation": "rank", ...}}'
```

Record cold start latency (`delayTime`) and execution time in benchmarks.

- [ ] **Step 8:** Set Convex env vars:

```bash
npx convex env set RUNPOD_ENDPOINT_ID <id>
npx convex env set RUNPOD_API_KEY <key>
```

- [ ] **Step 9:** Commit:

```bash
git add workers/runpod/
git commit -m "feat: runpod serverless worker"
```

---

### Task 11: Playlist workflow (Convex Workflow → RunPod `runsync`)

**Creates:** `convex/workflows.ts`, `convex/runpod.ts`, `convex/queries.ts`

**Why a workflow?** The playlist generation is multi-step (fetch weights → call RunPod → save result). If RunPod is slow or the server restarts, a plain action would fail silently. A **Convex Workflow** is durable — it retries failed steps automatically and picks up where it left off.

- [ ] **Step 1:** Write `convex/runpod.ts` (internal action that makes the actual HTTP call to RunPod):

```typescript
"use node";
import { internalAction } from "./_generated/server";
import { v } from "convex/values";

export const callRunsync = internalAction({
  args: {
    operation: v.string(),
    sessionTrackIds: v.array(v.string()),
    knownTrackIds: v.array(v.string()),
    context: v.any(),
    userWeightsBase64: v.string(),
  },
  handler: async (ctx, args) => {
    const endpointId = process.env.RUNPOD_ENDPOINT_ID!;
    const apiKey = process.env.RUNPOD_API_KEY!;
    const url = `https://api.runpod.ai/v2/${endpointId}/runsync?wait=120000`;

    const res = await fetch(url, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({ input: args }),
    });

    if (!res.ok) throw new Error(`RunPod ${res.status}: ${await res.text()}`);
    const body = await res.json();
    if (body.status !== "COMPLETED") throw new Error(`RunPod: ${body.status}`);
    return body.output;
  },
});
```

- [ ] **Step 2:** Write `convex/workflows.ts` (the durable workflow, see Part C for full code).

- [ ] **Step 3:** Write `convex/queries.ts`:

```typescript
import { internalQuery } from "./_generated/server";
import { v } from "convex/values";

export const getUserData = internalQuery({
  args: { userId: v.id("users") },
  handler: async (ctx, args) => {
    const weights = await ctx.db
      .query("userWeights")
      .withIndex("by_userId", (q) => q.eq("userId", args.userId))
      .order("desc")
      .first();
    // Fetch known track IDs from recent events
    const recentEvents = await ctx.db
      .query("playbackEvents")
      .withIndex("by_userId_time", (q) => q.eq("userId", args.userId))
      .order("desc")
      .take(200);
    const knownTrackIds = [...new Set(recentEvents.map((e) => e.spotifyTrackUri))];
    return {
      weightsBase64: weights?.weightsBase64 ?? "",
      knownTrackIds,
    };
  },
});
```

- [ ] **Step 4:** Test: start workflow from Convex dashboard → observe steps complete → get URIs back.

- [ ] **Step 5:** Commit:

```bash
git add convex/workflows.ts convex/runpod.ts convex/queries.ts
git commit -m "feat: playlist generation via Convex Workflow"
```

---

### Task 12: Online learning step (workflow)

**Creates:** Extension to `handler.py`, `convex/learning.ts` (workflow)

**Plain English:** After every skip/replay/completion, we want to nudge the User Brain a tiny bit. This is a workflow because: read weights from DB → call RunPod → save updated weights back. If RunPod is cold, the workflow waits and retries automatically.

- [ ] **Step 1:** Extend `handler.py` with `online_step` operation (already in Part C handler):
  - Input: `userWeightsBase64`, `events` (array of {song_features_26d, label_weight})
  - Process: load User Tower, one optimizer step (lr 1e-5, clip 0.3), serialize new weights
  - Output: `{ weightsBase64, version }`

- [ ] **Step 2:** Write `convex/learning.ts` as a workflow:

```typescript
import { workflow } from "./workflows";
import { internal } from "./_generated/api";
import { v } from "convex/values";

export const onlineLearn = workflow.define({
  args: {
    userId: v.id("users"),
    newEvent: v.object({ songFeatures: v.array(v.number()), labelWeight: v.number() }),
    replayEvents: v.array(v.object({ songFeatures: v.array(v.number()), labelWeight: v.number() })),
  },
  handler: async (step, args) => {
    // Step 1: Get current weights from DB
    const weightsDoc = await step.runQuery(internal.queries.getUserWeights, {
      userId: args.userId,
    });
    if (!weightsDoc) return;

    // Step 2: Call RunPod to do the actual math (with retry)
    const result = await step.runAction(
      internal.runpod.callRunsync,
      {
        operation: "online_step",
        userWeightsBase64: weightsDoc.weightsBase64,
        version: weightsDoc.version,
        events: [args.newEvent, ...args.replayEvents],
      },
      { retry: { maxAttempts: 2, initialBackoffMs: 500, base: 2 } },
    );

    // Step 3: Save updated weights back to DB
    if (result?.weightsBase64) {
      await step.runMutation(internal.mutations.updateUserWeights, {
        userId: args.userId,
        weightsBase64: result.weightsBase64,
        version: result.version,
      });
    }
  },
});
```

- [ ] **Step 3:** Unit test handler `online_step` on CPU:

```python
def test_online_step_updates_weights():
    import base64, io, torch
    from attune_ml.models.user_tower import UserTower

    model = UserTower()
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    original_b64 = base64.b64encode(buf.getvalue()).decode()

    event = {"input": {
        "operation": "online_step",
        "userWeightsBase64": original_b64,
        "version": 1,
        "events": [{"songFeatures": [0.5]*26, "labelWeight": 0.85}] * 9
    }}
    result = handler(event)
    assert result["version"] == 2
    assert result["weightsBase64"] != original_b64
```

- [ ] **Step 4:** Rebuild and redeploy Serverless image with updated handler.

- [ ] **Step 5:** Commit:

```bash
git add workers/runpod/handler.py convex/learning.ts
git commit -m "feat: online learning via Convex Workflow"
```

---

### Task 13: Context Encoder (**4090 Pod**, ~15–45 min synthetic)

**Creates:** `packages/attune_ml/attune_ml/models/context_encoder.py`, `packages/attune_ml/attune_ml/train/train_context.py`

- [ ] **Step 1:** Implement `context_encoder.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextEncoder(nn.Module):
    def __init__(self, user_dim: int = 128, context_dim: int = 16):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, user_dim),
            nn.Sigmoid(),
        )
        self.shift_net = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, user_dim),
        )

    def forward(self, user_vec: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gate = self.gate_net(context)        # (B, 128) values in [0, 1]
        shift = self.shift_net(context)      # (B, 128)
        out = user_vec + gate * shift        # gated residual
        return F.normalize(out, dim=-1)
```

- [ ] **Step 2:** Write test:

```python
def test_context_encoder_shape():
    from attune_ml.models.context_encoder import ContextEncoder
    model = ContextEncoder()
    user = torch.randn(4, 128)
    user = F.normalize(user, dim=-1)
    ctx = torch.randn(4, 16)
    out = model(user, ctx)
    assert out.shape == (4, 128)
    norms = torch.norm(out, dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)

def test_zero_context_preserves_user():
    from attune_ml.models.context_encoder import ContextEncoder
    model = ContextEncoder()
    model.eval()
    user = F.normalize(torch.randn(1, 128), dim=-1)
    ctx = torch.zeros(1, 16)
    out = model(user, ctx)
    # With zero context, gate should be ~0.5 (sigmoid(0)), shift ~0
    # Not exactly identity, but should be close if shift_net(0) ≈ 0
    # This is a soft check
    assert torch.cosine_similarity(user, out).item() > 0.8
```

- [ ] **Step 3:** Implement training script `train_context.py`:
  - Generates synthetic data: random user vectors, random contexts, synthetic "outcome" songs that should be closer to shifted user
  - Loss: triplet or InfoNCE between shifted_user and outcome_song (positive) vs random songs (negative)
  - Train ~50–200 epochs on synthetic pairs

- [ ] **Step 4:** Train on 4090 Pod:

```bash
python -m attune_ml.train.train_context --epochs 100 --output ml/export/context_encoder_v1.pt
```

Record wall time + loss in benchmarks.

- [ ] **Step 5:** Export as TorchScript, rebuild Serverless image:

```bash
docker build -t attune-worker:v2 -f workers/runpod/Dockerfile .
docker push <registry>/attune-worker:v2
# Update RunPod endpoint to new image
```

- [ ] **Step 6:** Set `ATTUNE_CONTEXT_ENABLED=1` in Serverless env.

- [ ] **Step 7:** Commit:

```bash
git add packages/attune_ml/attune_ml/models/context_encoder.py packages/attune_ml/attune_ml/train/train_context.py
git commit -m "feat: context encoder with gated residual"
```

---

### Task 14: GRU Sequencer (**4090 Pod**, ~10–30 min synthetic)

**Creates:** `packages/attune_ml/attune_ml/models/session_gru.py`, `packages/attune_ml/attune_ml/train/train_gru.py`

- [ ] **Step 1:** Implement `session_gru.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SessionGRU(nn.Module):
    def __init__(self, input_dim: int = 141, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, 141)
        Returns: (query (batch, 128), new_hidden (1, batch, 256))
        """
        out, hidden = self.gru(x, hidden)
        last = out[:, -1, :]  # (batch, 256)
        query = F.normalize(self.proj(last), dim=-1)
        return query, hidden

    def step(self, x: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Single step: x (batch, 141), hidden (1, batch, 256)"""
        x = x.unsqueeze(1)  # (batch, 1, 141)
        return self.forward(x, hidden)
```

- [ ] **Step 2:** Write test:

```python
def test_gru_forward():
    from attune_ml.models.session_gru import SessionGRU
    model = SessionGRU()
    x = torch.randn(2, 10, 141)  # 2 sessions, 10 steps each
    query, hidden = model(x)
    assert query.shape == (2, 128)
    assert hidden.shape == (1, 2, 256)
    norms = torch.norm(query, dim=-1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5)

def test_gru_step():
    from attune_ml.models.session_gru import SessionGRU
    model = SessionGRU()
    hidden = torch.zeros(1, 1, 256)
    x = torch.randn(1, 141)
    query, hidden = model.step(x, hidden)
    assert query.shape == (1, 128)
```

- [ ] **Step 3:** Implement `train_gru.py`:
  - Loss: predict next song embedding from GRU query (MSE or cosine loss between `query` and actual `next_song_embedding`)
  - Teacher forcing: at each step, feed actual song embedding (not predicted)
  - Synthetic sessions: random sequences of song embeddings with realistic transitions

- [ ] **Step 4:** Train on 4090 Pod:

```bash
python -m attune_ml.train.train_gru --epochs 50 --output ml/export/gru_v1.pt
```

- [ ] **Step 5:** Rebuild Serverless image with GRU, set `ATTUNE_GRU_ENABLED=1`.

- [ ] **Step 6:** Commit:

```bash
git add packages/attune_ml/attune_ml/models/session_gru.py packages/attune_ml/attune_ml/train/train_gru.py
git commit -m "feat: GRU sequencer for session coherence"
```

---

### Task 15: Expo app

**Creates:** Full Expo project in `app/`

- [ ] **Step 1:** Initialize Expo:

```bash
cd app
npx create-expo-app@latest . --template blank-typescript
npm install convex expo-auth-session expo-web-browser expo-location expo-crypto
```

- [ ] **Step 2:** Implement `app/hooks/useSpotifyAuth.ts`:

```typescript
import * as AuthSession from "expo-auth-session";
import { useAction } from "convex/react";
import { api } from "../convex/_generated/api";

const discovery = {
  authorizationEndpoint: "https://accounts.spotify.com/authorize",
  tokenEndpoint: "https://accounts.spotify.com/api/token",
};

export function useSpotifyAuth() {
  const linkSpotify = useAction(api.spotify.link);

  const [request, response, promptAsync] = AuthSession.useAuthRequest(
    {
      clientId: process.env.EXPO_PUBLIC_SPOTIFY_CLIENT_ID!,
      scopes: [
        "user-read-recently-played",
        "user-top-read",
        "user-library-read",
        "app-remote-control",
        "streaming",
      ],
      usePKCE: true,
      redirectUri: AuthSession.makeRedirectUri({ scheme: "attune" }),
    },
    discovery
  );

  const login = async () => {
    const result = await promptAsync();
    if (result.type === "success" && result.params.code) {
      const { userId } = await linkSpotify({
        code: result.params.code,
        redirectUri: request!.redirectUri,
      });
      return userId;
    }
    return null;
  };

  return { login, isReady: !!request };
}
```

- [ ] **Step 3:** Implement `app/hooks/useTelemetry.ts`:
  - Collect events in memory buffer
  - Every 5–15s or on significant event (skip/replay): flush batch to Convex
  - Attach context snapshot (time, location bucket, weather)
  - Generate idempotency key per batch (uuid)

- [ ] **Step 4:** Implement `app/hooks/usePlayback.ts`:
  - Platform detection: native vs web
  - Native: Spotify Remote SDK
  - Web: Spotify Web Playback SDK (`window.Spotify.Player`)
  - Unified interface: `play(uri)`, `pause()`, `skip()`, `getPosition()`, `onTrackChange(cb)`

- [ ] **Step 5:** Implement main screen (`app/app/(tabs)/index.tsx`):
  - Big "Attune this moment" button
  - Calls `useAction(api.playlist.generate)` with context + session tail + user weights
  - Shows loading state during RunPod inference
  - On success: queue tracks via playback hook

- [ ] **Step 6:** Implement now-playing strip:
  - Track name, artist, album art from Spotify
  - Progress ring (animated)
  - Skip / previous buttons

- [ ] **Step 7:** Test on physical iOS device:
  1. Login with Spotify
  2. Wait for onboarding (~1–5 min for cold train)
  3. Tap "Attune this moment"
  4. Verify tracks play
  5. Skip a track → verify event appears in Convex dashboard

- [ ] **Step 8:** Test on web:
  1. Same flow in browser
  2. Web Playback SDK creates player
  3. Verify playback and events

- [ ] **Step 9:** Commit:

```bash
git add app/
git commit -m "feat: expo app with spotify auth, playback, and telemetry"
```

---

### Task 16: Ship It deploy

**Creates:** production deployment across all services

- [ ] **Step 1:** Convex production deploy:

```bash
npx convex deploy
```

Set production env vars:
```bash
npx convex env set RUNPOD_ENDPOINT_ID <prod_id> --prod
npx convex env set RUNPOD_API_KEY <prod_key> --prod
npx convex env set SPOTIFY_CLIENT_ID <id> --prod
npx convex env set SPOTIFY_CLIENT_SECRET <secret> --prod
npx convex env set OPENWEATHER_API_KEY <key> --prod
```

- [ ] **Step 2:** RunPod — set 1 active worker for judging window:
  - Go to endpoint settings
  - Set Active Workers: 1
  - This eliminates cold start for judges (~$0.70/hr cost)

- [ ] **Step 3:** EAS build for iOS + Android:

```bash
cd app
eas build --profile preview --platform ios
eas build --profile preview --platform android
```

- [ ] **Step 4:** Web deploy:

```bash
cd app
npx expo export --platform web
# Deploy dist/ to Vercel/Netlify/EAS Hosting
```

- [ ] **Step 5:** e2e verification:

```bash
# RunPod health
curl "https://api.runpod.ai/v2/<ENDPOINT_ID>/health" \
  -H "authorization: Bearer <API_KEY>"
# Expected: {"workers": {"idle": 1, ...}}

# Convex smoke (from app or curl)
# Login → generate playlist → verify URIs returned
```

- [ ] **Step 6:** Create `docs/deploy.md` with all URLs, endpoint IDs, and commands.

- [ ] **Step 7:** After judging — turn off active workers:
  - Set Active Workers: 0
  - Set Idle Timeout: 5s (minimum cost)

- [ ] **Step 8:** Commit:

```bash
git add docs/deploy.md
git commit -m "chore: ship it deploy"
```

---

## Part F — Evaluation & metrics

| Metric | What it measures | How to compute | Target |
|--------|-----------------|----------------|--------|
| **Skip rate** | User satisfaction | Skips / total events after Attune playlist | <30% (vs ~50% random) |
| **Listen ratio** | Engagement | Mean listen_ratio across events | >0.7 |
| **Retrieval@10** | Model quality (offline) | Of held-out "liked" tracks, how many in top-10 FAISS results | >0.3 |
| **Retrieval@50** | Broader model quality | Same, top-50 | >0.6 |
| **GRU transition loss** | Session coherence | Cosine distance between GRU prediction and actual next song | <0.5 |
| **Cold start latency** | UX (first-time) | Time from code exchange to "ready" | <5 min |
| **Inference latency (warm)** | UX (playlist generation) | RunPod `executionTime` | <2s |
| **Inference latency (cold)** | UX worst case | RunPod `delayTime` + `executionTime` | <60s |
| **Online step latency** | Learning speed | Time for `online_step` runsync | <500ms warm |

### Benchmarks file template (`ml/evaluation/benchmarks.md`)

```markdown
# Attune Benchmarks

## Song Tower v1
- Date: 
- GPU: RTX 4090 24GB
- Batch: 512, Epochs: 30, Tau: 0.07, LR: 3e-4
- Final train loss: 
- Val loss: 
- Wall time: min

## FAISS Index
- Vectors: 114,000 × 128 float32
- Index type: IndexFlatIP
- Size on disk: MB
- Build time: s

## User Tower Cold
- Hardware: CPU 8 vCPU
- Epochs: 100
- Wall time: s

## Context Encoder v1
- GPU: RTX 4090
- Epochs: 
- Synthetic pairs: 
- Wall time: min

## GRU v1
- GPU: RTX 4090
- Epochs: 
- Sessions: 
- Wall time: min

## Serverless Inference
- Endpoint GPU: L4
- Cold start (first request): s
- Warm rank latency: ms
- Warm online_step latency: ms

## Evaluation (when available)
- Skip rate: %
- Mean listen ratio: 
- Retrieval@10: 
- Retrieval@50: 
```

---

## Execution

**Critical path:** Tasks 0–6 (ML ready) → Tasks 7–9 (backend) → Task 10–12 (inference live) → Task 15 (app) → Task 16 (ship).

**Parallelizable:**
- Tasks 7–8 (Convex) can run while Tasks 5–6 (GPU train) are in progress
- Task 15 (Expo app) can start UI work while waiting for Tasks 10–12

**Checkpoints (verify before continuing):**
1. After Task 6: FAISS index built, smoke query works
2. After Task 10: RunPod endpoint responds to `runsync`
3. After Task 11: Convex → RunPod → URIs returned end-to-end
4. After Task 15: Physical device plays music from Attune playlist
5. After Task 16: Judges can install and use the app

**Pick execution style:**
- **Subagent-driven:** One subagent per task, review between. Best for catching issues early.
- **Inline:** Batch execution with checkpoint pauses. Faster for experienced implementers.

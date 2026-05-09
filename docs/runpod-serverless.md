# RunPod Serverless — Attune inference

Official docs: [Serverless overview](https://docs.runpod.io/serverless/overview), [Send API requests](https://docs.runpod.io/serverless/endpoints/send-requests), [Operation reference](https://docs.runpod.io/serverless/endpoints/operation-reference), [Handler functions](https://docs.runpod.io/serverless/workers/handler-functions).

## Model: queue-based endpoint + Python handler

Attune ranks tracks with **PyTorch + FAISS** in a Docker worker. Use the **handler** pattern (`import runpod` + `runpod.serverless.start`) so RunPod exposes:

- `POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync` — wait for result (good for playlist latency; tune `?wait=` ms, default 90s max behavior per docs)
- `POST https://api.runpod.ai/v2/{ENDPOINT_ID}/run` — async + poll `/status/{jobId}` if jobs exceed sync limits
- `GET https://api.runpod.ai/v2/{ENDPOINT_ID}/health` — ops / e2e checks

**Request body (required shape):** top-level key **`input`** (object). Your handler reads `event["input"]`.

```json
{
  "input": {
    "operation": "rank",
    "sessionTrackIds": ["..."],
    "knownTrackIds": ["..."],
    "context": { "hourLocal": 22, "dow": 5, "locationBucket": "home", "weather": {} },
    "userWeightsBase64": "...",
    "spotifyAccessToken": "BQ..."
  }
}
```

**Auth:** `Authorization: Bearer <RUNPOD_API_KEY>` (see [API keys](https://docs.runpod.io/get-started/api-keys)). **Never** put this key in Expo — only **Convex** `process.env.RUNPOD_API_KEY` + `RUNPOD_ENDPOINT_ID`.

## Worker skeleton (rank)

```python
import base64
import runpod

def handler(event):
    inp = event.get("input") or {}
    op = inp.get("operation", "rank")
    if op == "rank":
        # load FAISS + SongTower + User + Context from disk baked in image
        # use inp["spotifyAccessToken"] only if worker must call Spotify; prefer Convex pre-fetch + pass features
        uris = []  # populate
        return {"uris": uris, "modelBundleId": "v1"}
    if op == "online_step":
        # optional: same worker image, different branch
        return {"weightsBase64": "...", "version": 2}
    return {"error": "unknown_operation"}

runpod.serverless.start({"handler": handler})
```

Load **large** artifacts (`faiss.index`, checkpoints) **inside the container image** or on a **network volume** attached to the template — warm once per worker to reduce cold start.

## Calling from Convex (correct place for the key)

Convex **action** (Node), not a public mutation:

```typescript
const url = `https://api.runpod.ai/v2/${process.env.RUNPOD_ENDPOINT_ID}/runsync?wait=120000`;
const res = await fetch(url, {
  method: "POST",
  headers: {
    "content-type": "application/json",
    authorization: `Bearer ${process.env.RUNPOD_API_KEY}`,
  },
  body: JSON.stringify({
    input: {
      operation: "rank",
      sessionTrackIds: args.sessionTrackIds,
      knownTrackIds: args.knownTrackIds,
      context: args.context,
      spotifyAccessToken: args.spotifyAccessToken, // short-lived from Convex internal refresh only
    },
  }),
});
if (!res.ok) throw new Error(await res.text());
const body = await res.json();
// runsync: completed output in body.output (see RunPod response shape in docs)
```

Parse the JSON body: completed **`runsync`** responses include an **`output`** field whose value is whatever your **handler `return`ed** (object or array). See [operation reference](https://docs.runpod.io/serverless/endpoints/operation-reference) example payloads.

**Authorization header:** docs show `authorization: <API_KEY>`; some snippets use `Bearer <API_KEY>`. Use the form that succeeds with your key (401 → try the other).

## Payload limits

- `/runsync`: up to **20 MB** input; result retention **~1 minute** (extend wait via query param per docs).
- `/run`: **10 MB** input; poll `/status`; results **~30 minutes**.

Prefer **passing track ids + context + optional precomputed user vector** over huge tensors; send **user weights** as compact base64 or omit if worker loads from S3 key passed in `input`.

## Cold starts (demo quality)

From [overview](https://docs.runpod.io/serverless/overview): cold start = container boot + **model load**. Mitigations in the RunPod console:

- **[Model caching](https://docs.runpod.io/serverless/endpoints/model-caching)** / template volumes  
- **[FlashBoot](https://docs.runpod.io/serverless/endpoints/endpoint-configurations#flashboot)**  
- **Min active workers** > 0 for judging window (costs idle $ — toggle after demo)

## Training vs inference

- **Song / Context / GRU offline trains:** use **GPU Pods** (multi-minute to multi-hour). See the **AI compute plan** table in [`docs/superpowers/plans/2026-05-09-attune-implementation-plan.md`](../superpowers/plans/2026-05-09-attune-implementation-plan.md) for **which GPU** and **wall-clock** targets.
- **FAISS build:** CPU lane.
- **Inference (`rank`, `online_step`):** this **Serverless** doc. GPU **flex/active $/s** by tier: [Endpoint GPU configuration](https://docs.runpod.io/serverless/endpoints/endpoint-configurations).

## Expo web + Spotify

Native **Remote** SDK is iOS/Android-first. For **web**, plan on [Spotify Web Playback SDK](https://developer.spotify.com/documentation/web-playback-sdk) + PKCE in the browser; keep **one** Convex contract for events so all platforms emit the same `playback_event_v1` shape.

## References

- [Handler functions](https://docs.runpod.io/serverless/workers/handler-functions) — errors, streaming  
- [Load balancing endpoints](https://docs.runpod.io/serverless/load-balancing/overview) — only if you need raw FastAPI routes instead of `input` wrapper

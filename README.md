# pi-cache-warmer

A [pi](https://github.com/mariozechner/pi) extension that keeps a provider's
**prompt cache** warm across idle periods in a session. It supports the
Anthropic Messages API and the OpenAI Chat Completions and Responses APIs.

## Install

```sh
pi install git:github.com/ejklock/pi-cache-warmer
```

pi loads `src/index.ts` directly (TypeScript at runtime — no build step). The extension has **zero runtime dependencies** (Node builtins + the global `fetch`); the `devDependencies` exist only for `npm run typecheck` and `npm test`, so there is nothing to compile after installing.

To disable it at any time without uninstalling, set `PI_CACHE_WARMER_DISABLED=1` (see [Configuration](#configuration)).

---

## The problem: a ~5-minute cache TTL

Anthropic and OpenAI both cache a request's prefix (system prompt, tools,
long context) and reuse it on subsequent requests to skip re-processing those
tokens — but only if a request reads that cached prefix within roughly 5 to
10 minutes of the last read. If a session sits idle for longer than that
(thinking, waiting on the user, a long-running background task), the cache
expires and the next real request pays full price to rebuild it.

## The mechanism: replay, don't rebuild

`pi-cache-warmer` captures the exact JSON body of the last real provider
request pi sent (via the `before_provider_request` extension event — the
request already carries the right cache markers, such as `cache_control` or
`prompt_cache_key`). When the agent goes idle, the extension arms a timer. If
nothing else happens before the timer fires, it **replays that exact
captured payload** against the same API the model uses, overriding only:

- the output-token cap (`max_tokens`, `max_completion_tokens`, or
  `max_output_tokens`, depending on the API) → `1`, or a small reasoning
  floor when the captured payload signals a reasoning request (the smallest
  cap a reasoning-capable model accepts)
- `stream` → `false` (no need for a streaming connection)
- `tool_choice` is dropped (irrelevant for a minimal reply)
- for the Responses API, `store` → `false`

Everything else — `system`/`input`, `messages`, `tools`, and their cache
markers — is sent byte-for-byte identical to the original request. Reading
that cached prefix resets its TTL, at effectively the cost of a small
output-token reply plus a cheap cache-read.

The extension **never rebuilds** a request from pi's internal `Context`
representation; v1 only replays what was already sent on the wire. It also
never mutates the real request: `before_provider_request` handlers return
`event.payload` completely unchanged.

## Scope

Three provider dialects are warmed, selected by the captured model's `api`:

- `anthropic-messages` — POSTs to `{baseUrl}/v1/messages`.
- `openai-completions` — POSTs to `{baseUrl}/chat/completions`.
- `openai-responses` — POSTs to `{baseUrl}/responses`.

Every other `api` (Google's `google-generative-ai`, etc.) is a no-op. They do
not schedule a warm or show warmer status. Those APIs have different (or no)
prompt-caching semantics that this extension does not attempt to model.

## Scheduling

- The warmer arms an idle timer on `agent_end` and cancels it as soon as new
  activity starts (`agent_start` or `input`). Cancellation removes the pending
  timer and its `next warm scheduled` footer text. An active warm reports its
  final outcome but does not retry after pi becomes busy.
- Another real provider request replaces the captured target. If it arrives
  during a warm, the warmer clears the old active footer and ignores the stale
  result. A normal later `agent_end` schedules the newly captured target.
- The timer is a `setTimeout(...).unref()`, so it never keeps the pi process
  alive on its own.
- Only one warm request is ever in flight at a time.
- After a successful or failed warm attempt, the timer re-arms automatically
  while the session remains idle.
- For eligible targets only (one of the three supported dialects), pi TUI
  and RPC UI modes show the persistent `pi-cache-warmer` footer status. It
  shows `next warm scheduled`,
  then `warming cache` during an attempt. After success, it shows
  `warmed (count N); next warm scheduled`, where `N` is the successful warm
  count for the current session.
- While pi remains idle, a non-success HTTP response shows `warm request failed
  (HTTP N); next warm scheduled`. Other request errors show `warm request
  failed; next warm scheduled`. If pi becomes busy before completion, the
  final outcome has no next-attempt suffix.
- The timer is cleared for good on `session_shutdown`.
- Installation is idempotent (guarded by a `Symbol.for` marker), so loading
  the extension twice in the same process is harmless.

## Configuration

| Env var | Default | Description |
| --- | --- | --- |
| `PI_CACHE_WARMER_DISABLED` | unset | Set to `1` to disable warming entirely. |
| `PI_CACHE_WARMER_INTERVAL_MS` | `240000` (4 min) | Idle delay before a warm request fires. Clamped to `[30000, 290000]` — always strictly under the 5-minute (`300000`) TTL. |

## Auth

The warm request resolves credentials **exclusively** through
`ctx.modelRegistry.getApiKeyAndHeaders(model)` — the same path pi itself uses.
This works transparently for both a static API key and an OAuth/subscription
login; the extension never reads environment variables or credential files
directly.

> **OAuth/subscription caveat:** if your Claude access is a Claude Pro/Max
> subscription via OAuth login (not a metered API key), each warm request
> still consumes a small amount of your usage allowance (a cache read plus a
> single output token). It is cheap, but it is not free. If that's a concern,
> set `PI_CACHE_WARMER_DISABLED=1` or raise `PI_CACHE_WARMER_INTERVAL_MS` to
> warm less often (at the cost of occasionally letting the cache expire).

## Cost

Each warm request costs roughly:

- 1 cache-read charge (Anthropic prices cache reads at a fraction — commonly
  cited around 0.1x — of the base input-token price) for the cached prefix, and
- 1 output token (`max_tokens: 1`).

For a typical system-prompt-plus-tools prefix this is a small fraction of the
cost of a full turn, traded against avoiding a full cache-miss rebuild the
next time the user actually sends a message.

## What this extension does *not* do

- It does not support Anthropic's 1-hour cache TTL beta — v1 targets the
  default 5-minute TTL only.
- It does not warm Google Gemini's `CachedContent` API — deferred.
- It does not warm any api outside the three supported dialects.
- It does not reconstruct or validate the payload against pi's `Context`
  model; it is a verbatim replay of the last request pi actually sent.

## Development

```bash
npm install
npm run typecheck
npm test
```

Tests run via Node's built-in test runner (`tsx --test src/*.test.ts`) and use
an injected `WarmerDeps` object (fake timers, a fetch spy, a stub
`resolveAuth`) — no real network I/O.

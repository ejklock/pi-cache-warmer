# pi-cache-warmer

A [pi](https://github.com/mariozechner/pi) extension that keeps the Anthropic/Claude
**prompt cache** warm across idle periods in a session.

## Install

```sh
pi install git:github.com/ejklock/pi-cache-warmer
```

pi loads `src/index.ts` directly (TypeScript at runtime — no build step). The extension has **zero runtime dependencies** (Node builtins + the global `fetch`); the `devDependencies` exist only for `npm run typecheck` and `npm test`, so there is nothing to compile after installing.

To disable it at any time without uninstalling, set `PI_CACHE_WARMER_DISABLED=1` (see [Configuration](#configuration)).

---

## The problem: a 5-minute cache TTL

Anthropic's prompt caching stores a cached prefix (system prompt, tools, long
context) and reuses it on subsequent requests to skip re-processing those
tokens — but only if a request reads that cached prefix **within 5 minutes**
of the last read. If a session sits idle for longer than that (thinking,
waiting on the user, a long-running background task), the cache expires and
the next real request pays full price to rebuild it.

## The mechanism: replay, don't rebuild

`pi-cache-warmer` captures the exact JSON body of the last real provider
request pi sent (via the `before_provider_request` extension event — the
request already carries the right `cache_control` breakpoints). When the
agent goes idle, the extension arms a timer. If nothing else happens before
the timer fires, it **replays that exact captured payload** against the
Anthropic Messages API, overriding only:

- `max_tokens` → `1` (the smallest possible response)
- `stream` → `false` (no need for a streaming connection)
- `tool_choice` is dropped (irrelevant for a 1-token reply)

Everything else — `system`, `messages`, `tools`, and their `cache_control`
markers — is sent byte-for-byte identical to the original request. Reading
that cached prefix resets its 5-minute TTL, at effectively the cost of a
single output token plus a cheap cache-read.

The extension **never rebuilds** a request from pi's internal `Context`
representation; v1 only replays what was already sent on the wire. It also
never mutates the real request: `before_provider_request` handlers return
`event.payload` completely unchanged.

## Scope

Only models with `api === "anthropic-messages"` are warmed. Every other
provider (OpenAI, Google, etc.) is a no-op — those APIs have different (or no)
prompt-caching semantics that this extension does not attempt to model.

## Scheduling

- The warmer arms an idle timer on `agent_end` and cancels it as soon as new
  activity starts (`agent_start`, `input`, or another real provider request).
- The timer is a `setTimeout(...).unref()`, so it never keeps the pi process
  alive on its own.
- Only one warm request is ever in flight at a time.
- After a successful (or failed) warm attempt, the timer re-arms automatically
  as long as the session is still idle.
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
- It does not warm any non-Anthropic provider.
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

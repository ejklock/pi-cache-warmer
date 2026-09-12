import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { setTimeout as delay } from "node:timers/promises";
import piCacheWarmer, {
	buildWarmBody,
	anthropicEndpoint,
	isAnthropicModel,
	isDisabled,
	resolveIntervalMs,
	createWarmer,
	type WarmerDeps,
} from "./index.ts";

interface FakeTimer {
	id: number;
	ms: number;
	cb: () => void;
}

interface FetchCall {
	url: string;
	init: { method: string; headers: Record<string, string>; body: string; signal: AbortSignal };
}

type FakeDeps = WarmerDeps & { timers: FakeTimer[]; fetchCalls: FetchCall[] };

function createFakeDeps(overrides: Partial<WarmerDeps> = {}): FakeDeps {
	const timers: FakeTimer[] = [];
	const fetchCalls: FetchCall[] = [];
	let nextId = 1;

	const deps: WarmerDeps = {
		setTimer: (cb, ms) => {
			const id = nextId++;
			timers.push({ id, ms, cb });
			return id;
		},
		clearTimer: (handle) => {
			const index = timers.findIndex((timer) => timer.id === handle);
			if (index >= 0) timers.splice(index, 1);
		},
		fetchImpl: async (url, init) => {
			fetchCalls.push({ url, init });
			return { ok: true, status: 200 };
		},
		resolveAuth: async () => ({ ok: true, apiKey: "test-api-key" }),
		isIdle: () => true,
		notify: () => {},
		setStatus: () => {},
		env: {},
		...overrides,
	};

	return Object.assign(deps, { timers, fetchCalls });
}

function anthropicModel(baseUrl = "https://api.anthropic.com") {
	return { api: "anthropic-messages", baseUrl } as unknown as Parameters<WarmerDeps["resolveAuth"]>[0];
}

function openAiModel() {
	return { api: "openai-completions", baseUrl: "https://api.openai.com" } as unknown as Parameters<
		WarmerDeps["resolveAuth"]
	>[0];
}

function samplePayload() {
	return {
		model: "claude-sonnet-4",
		max_tokens: 8192,
		stream: true,
		system: [{ type: "text", text: "system prompt", cache_control: { type: "ephemeral" } }],
		messages: [{ role: "user", content: "hello" }],
		tools: [{ name: "bash", description: "run a command" }],
		tool_choice: { type: "auto" },
	};
}

async function flushMicrotasks(): Promise<void> {
	await delay(0);
}

describe("resolveIntervalMs (AC3)", () => {
	it("defaults to 240000ms when unset", () => {
		assert.strictEqual(resolveIntervalMs({}), 240000);
	});

	it("parses a valid PI_CACHE_WARMER_INTERVAL_MS override", () => {
		assert.strictEqual(resolveIntervalMs({ PI_CACHE_WARMER_INTERVAL_MS: "60000" }), 60000);
	});

	it("clamps values at or above the 300000ms TTL to strictly below it", () => {
		const resolved = resolveIntervalMs({ PI_CACHE_WARMER_INTERVAL_MS: "600000" });
		assert.ok(resolved < 300000, `expected ${resolved} to be below 300000`);
	});

	it("clamps tiny values up to a sane lower bound", () => {
		assert.strictEqual(resolveIntervalMs({ PI_CACHE_WARMER_INTERVAL_MS: "10" }), 30000);
	});

	it("falls back to the default for a non-numeric override", () => {
		assert.strictEqual(resolveIntervalMs({ PI_CACHE_WARMER_INTERVAL_MS: "not-a-number" }), 240000);
	});
});

describe("isDisabled / isAnthropicModel / anthropicEndpoint", () => {
	it("isDisabled is true only for the exact string '1'", () => {
		assert.strictEqual(isDisabled({ PI_CACHE_WARMER_DISABLED: "1" }), true);
		assert.strictEqual(isDisabled({ PI_CACHE_WARMER_DISABLED: "true" }), false);
		assert.strictEqual(isDisabled({}), false);
	});

	it("isAnthropicModel matches only api === 'anthropic-messages'", () => {
		assert.strictEqual(isAnthropicModel(anthropicModel()), true);
		assert.strictEqual(isAnthropicModel(openAiModel()), false);
		assert.strictEqual(isAnthropicModel(undefined), false);
	});

	it("anthropicEndpoint appends /v1/messages and strips a trailing slash", () => {
		assert.strictEqual(anthropicEndpoint(anthropicModel()), "https://api.anthropic.com/v1/messages");
		const trailing = { baseUrl: "https://api.anthropic.com/" } as unknown as Parameters<typeof anthropicEndpoint>[0];
		assert.strictEqual(anthropicEndpoint(trailing), "https://api.anthropic.com/v1/messages");
	});
});

describe("buildWarmBody", () => {
	it("overrides max_tokens and stream without mutating the captured payload", () => {
		const payload = samplePayload();
		const body = buildWarmBody(payload) as ReturnType<typeof samplePayload>;

		assert.strictEqual(body.max_tokens, 1);
		assert.strictEqual(body.stream, false);
		assert.strictEqual(payload.max_tokens, 8192);
		assert.strictEqual(payload.stream, true);
		assert.strictEqual(body.messages, payload.messages);
		assert.strictEqual(body.system, payload.system);
		assert.strictEqual(body.tools, payload.tools);
	});

	it("excludes tool_choice from the serialized warm body", () => {
		const body = buildWarmBody(samplePayload());
		assert.strictEqual(JSON.stringify(body).includes("tool_choice"), false);
	});

	it("passes through non-object payloads unchanged", () => {
		assert.strictEqual(buildWarmBody(null), null);
		assert.strictEqual(buildWarmBody("raw"), "raw");
	});
});

describe("createWarmer — fireNow (AC1, AC2, AC4)", () => {
	it("AC1: issues exactly one request replaying system/messages/tools verbatim with max_tokens=1 and stream=false", async () => {
		const deps = createFakeDeps();
		const warmer = createWarmer(deps);
		const payload = samplePayload();

		warmer.capture(payload, anthropicModel());
		await warmer.fireNow();

		assert.strictEqual(deps.fetchCalls.length, 1);
		const [call] = deps.fetchCalls;
		assert.strictEqual(call!.url, "https://api.anthropic.com/v1/messages");

		const body = JSON.parse(call!.init.body) as ReturnType<typeof samplePayload>;
		assert.strictEqual(body.max_tokens, 1);
		assert.strictEqual(body.stream, false);
		assert.deepStrictEqual(body.system, payload.system);
		assert.deepStrictEqual(body.messages, payload.messages);
		assert.deepStrictEqual(body.tools, payload.tools);
		assert.strictEqual(warmer.warmCount, 1);
	});

	it("AC1: calls deps.notify exactly once with level 'info' and the running warm count on success", async () => {
		const notifications: { message: string; level: string }[] = [];
		const deps = createFakeDeps({
			notify: (message, level) => notifications.push({ message, level }),
		});
		const warmer = createWarmer(deps);
		warmer.capture(samplePayload(), anthropicModel());

		await warmer.fireNow();

		assert.strictEqual(notifications.length, 1);
		assert.strictEqual(notifications[0]!.level, "info");
		assert.ok(notifications[0]!.message.includes("warmed (count 1)"));
		assert.strictEqual(warmer.warmCount, 1);
	});

	it("shows active, successful, and next warm statuses without exposing the captured payload", async () => {
		const statuses: (string | undefined)[] = [];
		const deps = createFakeDeps();
		Object.assign(deps, { setStatus: (message: string | undefined) => statuses.push(message) });
		const warmer = createWarmer(deps);
		warmer.capture(samplePayload(), anthropicModel());

		await warmer.fireNow();

		assert.ok(statuses.some((message) => message?.includes("warming cache")));
		assert.ok(statuses.some((message) => message?.includes("warmed (count 1)")));
		assert.ok(statuses.some((message) => message?.includes("next warm")));
		assert.ok(statuses.every((message) => !message?.includes("system prompt") && !message?.includes("test-api-key")));
	});

	it("treats a non-success HTTP response as a failed attempt and schedules the next attempt", async () => {
		const statuses: (string | undefined)[] = [];
		const notifications: string[] = [];
		const deps = createFakeDeps({
			fetchImpl: async () => ({ ok: false, status: 503 }),
			notify: (message) => notifications.push(message),
		});
		Object.assign(deps, { setStatus: (message: string | undefined) => statuses.push(message) });
		const warmer = createWarmer(deps);
		warmer.capture(samplePayload(), anthropicModel());

		await warmer.fireNow();

		assert.strictEqual(warmer.warmCount, 0);
		assert.strictEqual(deps.timers.length, 1);
		assert.ok(notifications.some((message) => message.includes("HTTP 503")));
		assert.ok(statuses.some((message) => message?.includes("HTTP 503") && message.includes("next warm")));
	});

	it("AC4: resolves auth via the injected resolveAuth and sends it as x-api-key when no authorization header is present", async () => {
		const deps = createFakeDeps({
			resolveAuth: async (model) => {
				assert.strictEqual(model.api, "anthropic-messages");
				return { ok: true, apiKey: "resolved-key" };
			},
		});
		const warmer = createWarmer(deps);
		warmer.capture(samplePayload(), anthropicModel());

		await warmer.fireNow();

		const [call] = deps.fetchCalls;
		assert.strictEqual(call!.init.headers["x-api-key"], "resolved-key");
		assert.strictEqual(call!.init.headers["anthropic-version"], "2023-06-01");
	});

	it("AC4: prefers an OAuth authorization header over x-api-key when resolveAuth returns one", async () => {
		const deps = createFakeDeps({
			resolveAuth: async () => ({ ok: true, headers: { authorization: "Bearer oauth-token" } }),
		});
		const warmer = createWarmer(deps);
		warmer.capture(samplePayload(), anthropicModel());

		await warmer.fireNow();

		const [call] = deps.fetchCalls;
		assert.strictEqual(call!.init.headers.authorization, "Bearer oauth-token");
		assert.strictEqual("x-api-key" in call!.init.headers, false);
	});

	it("AC2: issues no request when disabled via PI_CACHE_WARMER_DISABLED=1", async () => {
		const deps = createFakeDeps({ env: { PI_CACHE_WARMER_DISABLED: "1" } });
		const warmer = createWarmer(deps);
		warmer.capture(samplePayload(), anthropicModel());

		await warmer.fireNow();

		assert.strictEqual(deps.fetchCalls.length, 0);
	});

	it("AC2: issues no request for a non-Anthropic model", async () => {
		const deps = createFakeDeps();
		const warmer = createWarmer(deps);
		warmer.capture(samplePayload(), openAiModel());

		await warmer.fireNow();

		assert.strictEqual(deps.fetchCalls.length, 0);
	});

	it("AC2: issues no request when the session is not idle", async () => {
		const deps = createFakeDeps({ isIdle: () => false });
		const warmer = createWarmer(deps);
		warmer.capture(samplePayload(), anthropicModel());

		await warmer.fireNow();

		assert.strictEqual(deps.fetchCalls.length, 0);
	});

	it("AC2: issues no request when no payload has been captured yet", async () => {
		const deps = createFakeDeps();
		const warmer = createWarmer(deps);

		await warmer.fireNow();

		assert.strictEqual(deps.fetchCalls.length, 0);
	});

	it("skips an auth failure without exposing credential details", async () => {
		const notifications: string[] = [];
		const deps = createFakeDeps({
			resolveAuth: async () => ({ ok: false, error: "no credentials configured: test-api-key" }),
			notify: (message) => notifications.push(message),
		});
		const warmer = createWarmer(deps);
		warmer.capture(samplePayload(), anthropicModel());

		await assert.doesNotReject(() => warmer.fireNow());

		assert.strictEqual(deps.fetchCalls.length, 0);
		assert.ok(notifications.some((message) => message.includes("warm request skipped")));
		assert.ok(notifications.every((message) => !message.includes("test-api-key")));
	});

	it("handles a network failure without exposing network details", async () => {
		const notifications: string[] = [];
		const deps = createFakeDeps({
			fetchImpl: async () => {
				throw new Error("ECONNRESET test-api-key");
			},
			notify: (message) => notifications.push(message),
		});
		const warmer = createWarmer(deps);
		warmer.capture(samplePayload(), anthropicModel());

		await assert.doesNotReject(() => warmer.fireNow());

		assert.ok(notifications.some((message) => message.includes("warm request failed")));
		assert.ok(notifications.every((message) => !message.includes("ECONNRESET") && !message.includes("test-api-key")));
	});
});

describe("createWarmer — scheduler (AC5)", () => {
	it("arm() schedules a single unref-able timer for resolveIntervalMs(env)", () => {
		const deps = createFakeDeps({ env: { PI_CACHE_WARMER_INTERVAL_MS: "50000" } });
		const warmer = createWarmer(deps);
		warmer.capture(samplePayload(), anthropicModel());

		warmer.arm();

		assert.strictEqual(deps.timers.length, 1);
		assert.strictEqual(deps.timers[0]!.ms, 50000);
	});

	it("does not schedule or advertise a warm when disabled", () => {
		const statuses: (string | undefined)[] = [];
		const deps = createFakeDeps({ env: { PI_CACHE_WARMER_DISABLED: "1" }, setStatus: (message) => statuses.push(message) });
		const warmer = createWarmer(deps);
		warmer.capture(samplePayload(), anthropicModel());

		warmer.arm();

		assert.strictEqual(deps.timers.length, 0);
		assert.ok(statuses.every((message) => !message?.includes("next warm scheduled")));
		assert.strictEqual(statuses.at(-1), undefined);
	});

	it("does not schedule or advertise a warm for an unsupported model", () => {
		const statuses: (string | undefined)[] = [];
		const deps = createFakeDeps({ setStatus: (message) => statuses.push(message) });
		const warmer = createWarmer(deps);
		warmer.capture(samplePayload(), openAiModel());

		warmer.arm();

		assert.strictEqual(deps.timers.length, 0);
		assert.ok(statuses.every((message) => !message?.includes("next warm scheduled")));
		assert.strictEqual(statuses.at(-1), undefined);
	});

	it("cancel() clears the pending timer and scheduled status", () => {
		const statuses: (string | undefined)[] = [];
		const deps = createFakeDeps({ setStatus: (message) => statuses.push(message) });
		const warmer = createWarmer(deps);
		warmer.capture(samplePayload(), anthropicModel());

		warmer.arm();
		warmer.cancel();

		assert.strictEqual(deps.timers.length, 0);
		assert.strictEqual(statuses.at(-1), undefined);
	});

	it("re-arming replaces rather than stacks the pending timer", () => {
		const deps = createFakeDeps();
		const warmer = createWarmer(deps);
		warmer.capture(samplePayload(), anthropicModel());

		warmer.arm();
		warmer.arm();

		assert.strictEqual(deps.timers.length, 1);
	});

	it("dispose() clears the pending timer and prevents any further arming", () => {
		const deps = createFakeDeps();
		const warmer = createWarmer(deps);
		warmer.capture(samplePayload(), anthropicModel());

		warmer.arm();
		warmer.dispose();

		assert.strictEqual(deps.timers.length, 0);

		warmer.arm();
		assert.strictEqual(deps.timers.length, 0);
	});

	it("invalidates a captured target after model selection until the next capture", async () => {
		let resolveAuth: (auth: Awaited<ReturnType<WarmerDeps["resolveAuth"]>>) => void = () => {};
		const auth = new Promise<Awaited<ReturnType<WarmerDeps["resolveAuth"]>>>((resolve) => {
			resolveAuth = resolve;
		});
		const deps = createFakeDeps({ resolveAuth: async () => auth });
		const warmer = createWarmer(deps);
		const firstModel = anthropicModel("https://first.example");
		const selectedModel = anthropicModel("https://selected.example");

		warmer.capture(samplePayload(), firstModel);
		warmer.arm();
		const firstAttempt = warmer.fireNow();
		warmer.updateModel(selectedModel);
		resolveAuth({ ok: true, apiKey: "test-api-key" });
		await firstAttempt;

		assert.strictEqual(deps.timers.length, 0);
		assert.strictEqual(deps.fetchCalls.length, 0);

		warmer.capture(samplePayload(), selectedModel);
		await warmer.fireNow();

		assert.strictEqual(deps.fetchCalls.length, 1);
		assert.strictEqual(deps.fetchCalls[0]!.url, "https://selected.example/v1/messages");
	});

	it("prevents a fetch when disposed before authentication resolves", async () => {
		let resolveAuth: (auth: Awaited<ReturnType<WarmerDeps["resolveAuth"]>>) => void = () => {};
		const auth = new Promise<Awaited<ReturnType<WarmerDeps["resolveAuth"]>>>((resolve) => {
			resolveAuth = resolve;
		});
		const deps = createFakeDeps({ resolveAuth: async () => auth });
		const warmer = createWarmer(deps);
		warmer.capture(samplePayload(), anthropicModel());

		const attempt = warmer.fireNow();
		warmer.dispose();
		resolveAuth({ ok: true, apiKey: "test-api-key" });
		await attempt;

		assert.strictEqual(deps.fetchCalls.length, 0);
		assert.strictEqual(deps.timers.length, 0);
		assert.strictEqual(warmer.warmCount, 0);
	});

	it("aborts an active fetch and prevents later warm success, status, and retry", async () => {
		let resolveFetch: (response: Awaited<ReturnType<WarmerDeps["fetchImpl"]>>) => void = () => {};
		const fetchResult = new Promise<Awaited<ReturnType<WarmerDeps["fetchImpl"]>>>((resolve) => {
			resolveFetch = resolve;
		});
		const statuses: (string | undefined)[] = [];
		const notifications: string[] = [];
		let requestSignal: AbortSignal | undefined;
		const deps = createFakeDeps({
			fetchImpl: async (_url, init) => {
				requestSignal = init.signal;
				return fetchResult;
			},
			setStatus: (message) => statuses.push(message),
			notify: (message) => notifications.push(message),
		});
		const warmer = createWarmer(deps);
		warmer.capture(samplePayload(), anthropicModel());

		const attempt = warmer.fireNow();
		await flushMicrotasks();
		warmer.dispose();

		assert.strictEqual(requestSignal?.aborted, true);
		resolveFetch({ ok: true, status: 200 });
		await attempt;

		assert.strictEqual(warmer.warmCount, 0);
		assert.strictEqual(deps.timers.length, 0);
		assert.deepStrictEqual(notifications, []);
		assert.strictEqual(statuses.at(-1), undefined);
		assert.ok(statuses.every((message) => !message?.includes("warmed") && !message?.includes("next warm scheduled")));
	});

	it("re-arms after a successful warm while the session is still idle", async () => {
		const deps = createFakeDeps();
		const warmer = createWarmer(deps);
		warmer.capture(samplePayload(), anthropicModel());

		warmer.arm();
		const scheduled = deps.timers[0]!;
		deps.timers.length = 0;
		scheduled.cb();
		await flushMicrotasks();

		assert.strictEqual(deps.fetchCalls.length, 1);
		assert.strictEqual(deps.timers.length, 1);
	});

	it("does not accept a stale fetch result after a replacement target is captured (AC8)", async () => {
		let resolveFetch: (response: Awaited<ReturnType<WarmerDeps["fetchImpl"]>>) => void = () => {};
		const fetchResult = new Promise<Awaited<ReturnType<WarmerDeps["fetchImpl"]>>>((resolve) => {
			resolveFetch = resolve;
		});
		const statuses: (string | undefined)[] = [];
		let currentFooter: string | undefined;
		const notifications: string[] = [];
		const deps = createFakeDeps({
			fetchImpl: async () => fetchResult,
			setStatus: (message) => {
				statuses.push(message);
				currentFooter = message;
			},
			notify: (message) => notifications.push(message),
		});
		const warmer = createWarmer(deps);
		warmer.capture(samplePayload(), anthropicModel());

		const attempt = warmer.fireNow();
		await flushMicrotasks();
		assert.strictEqual(currentFooter, "warming cache");
		warmer.capture({ ...samplePayload(), model: "replacement" }, anthropicModel());
		assert.strictEqual(currentFooter, undefined);
		resolveFetch({ ok: true, status: 200 });
		await attempt;

		assert.strictEqual(warmer.warmCount, 0);
		assert.strictEqual(deps.timers.length, 0);
		assert.deepStrictEqual(statuses, ["warming cache", undefined]);
		assert.strictEqual(currentFooter, undefined);
		assert.deepStrictEqual(notifications, []);
	});

	it("schedules a replacement target when agent_end occurs during a stale in-flight warm", async () => {
		let resolveFetch: (response: Awaited<ReturnType<WarmerDeps["fetchImpl"]>>) => void = () => {};
		const fetchResult = new Promise<Awaited<ReturnType<WarmerDeps["fetchImpl"]>>>((resolve) => {
			resolveFetch = resolve;
		});
		const statuses: (string | undefined)[] = [];
		const notifications: string[] = [];
		const deps = createFakeDeps({
			fetchImpl: async () => fetchResult,
			setStatus: (message) => statuses.push(message),
			notify: (message) => notifications.push(message),
		});
		const warmer = createWarmer(deps);
		warmer.capture(samplePayload(), anthropicModel());

		const staleAttempt = warmer.fireNow();
		await flushMicrotasks();
		warmer.capture({ ...samplePayload(), model: "replacement" }, anthropicModel());
		warmer.arm();
		resolveFetch({ ok: true, status: 200 });
		await staleAttempt;

		assert.strictEqual(warmer.warmCount, 0);
		assert.deepStrictEqual(notifications, []);
		assert.strictEqual(deps.timers.length, 1);
		assert.strictEqual(deps.timers[0]!.ms, 240000);
		assert.strictEqual(statuses.at(-1), "next warm scheduled");
	});

	it("shows a settled warm result but does not retry after cancellation during an active request", async () => {
		let idle = true;
		let resolveFetch: (response: Awaited<ReturnType<WarmerDeps["fetchImpl"]>>) => void = () => {};
		const fetchResult = new Promise<Awaited<ReturnType<WarmerDeps["fetchImpl"]>>>((resolve) => {
			resolveFetch = resolve;
		});
		const statuses: (string | undefined)[] = [];
		const notifications: string[] = [];
		const deps = createFakeDeps({
			isIdle: () => idle,
			fetchImpl: async () => fetchResult,
			setStatus: (message) => statuses.push(message),
			notify: (message) => notifications.push(message),
		});
		const warmer = createWarmer(deps);
		warmer.capture(samplePayload(), anthropicModel());

		const attempt = warmer.fireNow();
		await flushMicrotasks();
		warmer.cancel();
		idle = false;
		idle = true;
		resolveFetch({ ok: true, status: 200 });
		await attempt;

		assert.strictEqual(deps.timers.length, 0);
		assert.strictEqual(statuses.at(-1), "warmed (count 1)");
		assert.ok(notifications.some((message) => message === "warmed (count 1)"));
		assert.ok(statuses.every((message) => !message?.includes("next warm scheduled")));
	});

	it("retains one future attempt when a scheduled warm finds Pi busy", async () => {
		let idle = false;
		const deps = createFakeDeps({ isIdle: () => idle });
		const warmer = createWarmer(deps);
		warmer.capture(samplePayload(), anthropicModel());
		warmer.arm();

		const busyAttempt = deps.timers.shift()!;
		busyAttempt.cb();
		await flushMicrotasks();

		assert.strictEqual(deps.fetchCalls.length, 0);
		assert.strictEqual(deps.timers.length, 1);

		idle = true;
		const eligibleAttempt = deps.timers.shift()!;
		eligibleAttempt.cb();
		await flushMicrotasks();

		assert.strictEqual(deps.fetchCalls.length, 1);
		assert.strictEqual(deps.timers.length, 1);
	});

	it("does not re-arm once disposed even if the timer fires", async () => {
		const deps = createFakeDeps();
		const warmer = createWarmer(deps);
		warmer.capture(samplePayload(), anthropicModel());

		warmer.arm();
		const scheduled = deps.timers[0]!;
		deps.timers.length = 0;
		warmer.dispose();
		scheduled.cb();
		await flushMicrotasks();

		assert.strictEqual(deps.fetchCalls.length, 0);
		assert.strictEqual(deps.timers.length, 0);
	});
});

describe("default extension wiring", () => {
	it("updates persistent status only when the extension context has a UI", () => {
		const handlers = new Map<string, (event: { payload?: unknown }, context: unknown) => unknown>();
		const pi = {
			on(event: string, handler: (payload: { payload?: unknown }, context: unknown) => unknown) {
				handlers.set(event, handler);
			},
		};
		const uiStatuses: { key: string; message: string | undefined }[] = [];
		const context = {
			model: anthropicModel(),
			hasUI: true,
			isIdle: () => true,
			ui: { setStatus: (key: string, message: string | undefined) => uiStatuses.push({ key, message }) },
			modelRegistry: { getApiKeyAndHeaders: async () => ({ ok: true, apiKey: "test-api-key" }) },
		};
		piCacheWarmer(pi as never);

		const payload = samplePayload();
		assert.strictEqual(handlers.get("before_provider_request")!({ payload }, context), payload);
		handlers.get("agent_end")!({}, context);

		assert.deepStrictEqual(uiStatuses.at(-1), { key: "pi-cache-warmer", message: "next warm scheduled" });

		context.hasUI = false;
		handlers.get("agent_end")!({}, context);
		assert.strictEqual(uiStatuses.length, 1);
	});
});

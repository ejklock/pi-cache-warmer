import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { setTimeout as delay } from "node:timers/promises";
import piCacheWarmer, {
	getDialect,
	isWarmableModel,
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

function openAiModel(baseUrl = "https://api.openai.com/v1") {
	return { api: "openai-completions", baseUrl } as unknown as Parameters<WarmerDeps["resolveAuth"]>[0];
}

function openAiResponsesModel(baseUrl = "https://api.openai.com/v1") {
	return { api: "openai-responses", baseUrl } as unknown as Parameters<WarmerDeps["resolveAuth"]>[0];
}

function geminiModel() {
	return { api: "google-generative-ai", baseUrl: "https://generativelanguage.googleapis.com" } as unknown as Parameters<
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

describe("isDisabled / isWarmableModel / getDialect", () => {
	it("isDisabled is true only for the exact string '1'", () => {
		assert.strictEqual(isDisabled({ PI_CACHE_WARMER_DISABLED: "1" }), true);
		assert.strictEqual(isDisabled({ PI_CACHE_WARMER_DISABLED: "true" }), false);
		assert.strictEqual(isDisabled({}), false);
	});

	it("isWarmableModel matches the anthropic-messages and openai dialects", () => {
		assert.strictEqual(isWarmableModel(anthropicModel()), true);
		assert.strictEqual(isWarmableModel(openAiModel()), true);
		assert.strictEqual(isWarmableModel(openAiResponsesModel()), true);
		assert.strictEqual(isWarmableModel(undefined), false);
	});

	it("AC5: isWarmableModel rejects an unsupported api such as google-generative-ai", () => {
		assert.strictEqual(isWarmableModel(geminiModel()), false);
		assert.strictEqual(getDialect(geminiModel()), undefined);
	});

	it("anthropic-messages dialect endpoint appends /v1/messages and strips a trailing slash", () => {
		const dialect = getDialect(anthropicModel())!;
		assert.strictEqual(dialect.endpoint(anthropicModel()), "https://api.anthropic.com/v1/messages");
		const trailing = anthropicModel("https://api.anthropic.com/");
		assert.strictEqual(dialect.endpoint(trailing), "https://api.anthropic.com/v1/messages");
	});

	it("AC2: openai-completions dialect endpoint appends /chat/completions to a baseUrl that already includes /v1", () => {
		const dialect = getDialect(openAiModel())!;
		assert.strictEqual(dialect.endpoint(openAiModel()), "https://api.openai.com/v1/chat/completions");
	});

	it("AC3: openai-responses dialect endpoint appends /responses to a baseUrl that already includes /v1", () => {
		const dialect = getDialect(openAiResponsesModel())!;
		assert.strictEqual(dialect.endpoint(openAiResponsesModel()), "https://api.openai.com/v1/responses");
	});
});

describe("minimizeBody per dialect (AC1, AC2, AC3, AC4, AC7)", () => {
	it("AC1/AC7: anthropic dialect overrides max_tokens and stream without mutating the captured payload", () => {
		const payload = samplePayload();
		const dialect = getDialect(anthropicModel())!;
		const body = dialect.minimizeBody(payload) as ReturnType<typeof samplePayload>;

		assert.strictEqual(body.max_tokens, 1);
		assert.strictEqual(body.stream, false);
		assert.strictEqual(payload.max_tokens, 8192);
		assert.strictEqual(payload.stream, true);
		assert.strictEqual(body.messages, payload.messages);
		assert.strictEqual(body.system, payload.system);
		assert.strictEqual(body.tools, payload.tools);
	});

	it("AC1: excludes tool_choice from the serialized anthropic warm body", () => {
		const dialect = getDialect(anthropicModel())!;
		const body = dialect.minimizeBody(samplePayload());
		assert.strictEqual(JSON.stringify(body).includes("tool_choice"), false);
	});

	it("passes through non-object payloads unchanged for every dialect", () => {
		for (const model of [anthropicModel(), openAiModel(), openAiResponsesModel()]) {
			const dialect = getDialect(model)!;
			assert.strictEqual(dialect.minimizeBody(null), null);
			assert.strictEqual(dialect.minimizeBody("raw"), "raw");
		}
	});

	it("AC2: openai-completions dialect minimizes max_tokens to 1 without a reasoning signal", () => {
		const payload = { model: "gpt-4o", max_tokens: 4096, stream: true, messages: [{ role: "user", content: "hi" }] };
		const dialect = getDialect(openAiModel())!;
		const body = dialect.minimizeBody(payload) as typeof payload;

		assert.strictEqual(body.max_tokens, 1);
		assert.strictEqual(body.stream, false);
		assert.strictEqual(payload.max_tokens, 4096);
	});

	it("AC2: openai-completions dialect minimizes whichever of max_tokens/max_completion_tokens is present", () => {
		const payload = { model: "gpt-4o", max_completion_tokens: 4096, stream: true, messages: [] };
		const dialect = getDialect(openAiModel())!;
		const body = dialect.minimizeBody(payload) as Record<string, unknown>;

		assert.strictEqual(body.max_completion_tokens, 1);
		assert.strictEqual("max_tokens" in body, false);
	});

	it("AC4: openai-completions dialect raises the cap to the reasoning floor when reasoning_effort is present", () => {
		const payload = { model: "o1", max_completion_tokens: 4096, reasoning_effort: "medium", stream: true, messages: [] };
		const dialect = getDialect(openAiModel())!;
		const body = dialect.minimizeBody(payload) as Record<string, unknown>;

		assert.strictEqual(body.max_completion_tokens, 16);
		assert.ok((body.max_completion_tokens as number) > 1);
	});

	it("AC3: openai-responses dialect minimizes max_output_tokens to 1 and keeps store false without a reasoning signal", () => {
		const payload = { model: "gpt-4o", max_output_tokens: 4096, stream: true, input: [{ role: "user", content: "hi" }] };
		const dialect = getDialect(openAiResponsesModel())!;
		const body = dialect.minimizeBody(payload) as Record<string, unknown>;

		assert.strictEqual(body.max_output_tokens, 1);
		assert.strictEqual(body.store, false);
		assert.strictEqual(body.stream, false);
		assert.strictEqual(body.input, payload.input);
	});

	it("AC4: openai-responses dialect raises the cap to the reasoning floor when a reasoning object is present", () => {
		const payload = {
			model: "o1",
			max_output_tokens: 4096,
			reasoning: { effort: "medium" },
			stream: true,
			input: [],
		};
		const dialect = getDialect(openAiResponsesModel())!;
		const body = dialect.minimizeBody(payload) as Record<string, unknown>;

		assert.strictEqual(body.max_output_tokens, 16);
		assert.ok((body.max_output_tokens as number) > 1);
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

	it("AC5: issues no request for an unsupported api such as google-generative-ai", async () => {
		const deps = createFakeDeps();
		const warmer = createWarmer(deps);
		warmer.capture(samplePayload(), geminiModel());

		await warmer.fireNow();

		assert.strictEqual(deps.fetchCalls.length, 0);
	});

	it("AC2: warms an openai-completions model by POSTing to /chat/completions with no anthropic-version header", async () => {
		const deps = createFakeDeps();
		const warmer = createWarmer(deps);
		const payload = {
			model: "gpt-4o",
			max_tokens: 4096,
			stream: true,
			system: "system prompt",
			messages: [{ role: "user", content: "hello" }],
			tools: [{ name: "bash" }],
			tool_choice: "auto",
			prompt_cache_key: "session-1",
		};

		warmer.capture(payload, openAiModel());
		await warmer.fireNow();

		assert.strictEqual(deps.fetchCalls.length, 1);
		const [call] = deps.fetchCalls;
		assert.strictEqual(call!.url, "https://api.openai.com/v1/chat/completions");
		assert.strictEqual("anthropic-version" in call!.init.headers, false);
		assert.strictEqual(call!.init.headers.authorization, "Bearer test-api-key");

		const body = JSON.parse(call!.init.body) as typeof payload;
		assert.strictEqual(body.max_tokens, 1);
		assert.strictEqual(body.stream, false);
		assert.deepStrictEqual(body.messages, payload.messages);
		assert.deepStrictEqual(body.tools, payload.tools);
		assert.strictEqual(body.system, payload.system);
		assert.strictEqual(body.prompt_cache_key, payload.prompt_cache_key);
		assert.strictEqual(JSON.stringify(body).includes("tool_choice"), false);
		assert.strictEqual(warmer.warmCount, 1);
	});

	it("AC4: warms an openai-completions reasoning model at the reasoning floor instead of 1", async () => {
		const deps = createFakeDeps();
		const warmer = createWarmer(deps);
		const payload = {
			model: "o1",
			max_completion_tokens: 4096,
			reasoning_effort: "medium",
			stream: true,
			messages: [{ role: "user", content: "hello" }],
		};

		warmer.capture(payload, openAiModel());
		await warmer.fireNow();

		const [call] = deps.fetchCalls;
		const body = JSON.parse(call!.init.body) as Record<string, unknown>;
		assert.strictEqual(body.max_completion_tokens, 16);
	});

	it("AC3: warms an openai-responses model by POSTing to /responses with store:false and no anthropic-version header", async () => {
		const deps = createFakeDeps();
		const warmer = createWarmer(deps);
		const payload = {
			model: "gpt-4o",
			max_output_tokens: 4096,
			stream: true,
			input: [{ role: "user", content: "hello" }],
			tools: [{ name: "bash" }],
		};

		warmer.capture(payload, openAiResponsesModel());
		await warmer.fireNow();

		assert.strictEqual(deps.fetchCalls.length, 1);
		const [call] = deps.fetchCalls;
		assert.strictEqual(call!.url, "https://api.openai.com/v1/responses");
		assert.strictEqual("anthropic-version" in call!.init.headers, false);
		assert.strictEqual(call!.init.headers.authorization, "Bearer test-api-key");

		const body = JSON.parse(call!.init.body) as typeof payload & { store: boolean };
		assert.strictEqual(body.max_output_tokens, 1);
		assert.strictEqual(body.store, false);
		assert.strictEqual(body.stream, false);
		assert.deepStrictEqual(body.input, payload.input);
		assert.deepStrictEqual(body.tools, payload.tools);
		assert.strictEqual(warmer.warmCount, 1);
	});

	it("AC4: warms an openai-responses reasoning model at the reasoning floor instead of 1", async () => {
		const deps = createFakeDeps();
		const warmer = createWarmer(deps);
		const payload = {
			model: "o1",
			max_output_tokens: 4096,
			reasoning: { effort: "medium" },
			stream: true,
			input: [{ role: "user", content: "hello" }],
		};

		warmer.capture(payload, openAiResponsesModel());
		await warmer.fireNow();

		const [call] = deps.fetchCalls;
		const body = JSON.parse(call!.init.body) as Record<string, unknown>;
		assert.strictEqual(body.max_output_tokens, 16);
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

	it("AC5: does not schedule or advertise a warm for an unsupported api such as google-generative-ai", () => {
		const statuses: (string | undefined)[] = [];
		const deps = createFakeDeps({ setStatus: (message) => statuses.push(message) });
		const warmer = createWarmer(deps);
		warmer.capture(samplePayload(), geminiModel());

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

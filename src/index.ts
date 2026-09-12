import type { ExtensionAPI, ExtensionContext } from "@mariozechner/pi-coding-agent";

const INSTALL_SYMBOL = Symbol.for("klock.pi-cache-warmer.installed");

const ANTHROPIC_API = "anthropic-messages";
const OPENAI_COMPLETIONS_API = "openai-completions";
const OPENAI_RESPONSES_API = "openai-responses";
const ANTHROPIC_VERSION = "2023-06-01";
// A minimized reasoning request still needs enough output tokens to emit at least
// one reasoning-then-answer step; 1 token is rejected by reasoning-capable models.
const REASONING_OUTPUT_FLOOR = 16;
const DISABLE_ENV_VAR = "PI_CACHE_WARMER_DISABLED";
const INTERVAL_ENV_VAR = "PI_CACHE_WARMER_INTERVAL_MS";
const DEFAULT_INTERVAL_MS = 240000;
const MIN_INTERVAL_MS = 30000;
// Anthropic's prompt-cache TTL is 300000ms; staying strictly under it guarantees
// the warm request lands before the cached prefix would otherwise expire.
const MAX_INTERVAL_MS = 290000;
const WARM_REQUEST_TIMEOUT_MS = 15000;

type AnyModel = NonNullable<ExtensionContext["model"]>;
type ResolveAuthFn = ExtensionContext["modelRegistry"]["getApiKeyAndHeaders"];
type ResolvedAuth = Awaited<ReturnType<ResolveAuthFn>>;
type AuthorizedAuth = Extract<ResolvedAuth, { ok: true }>;
type EnvRecord = Record<string, string | undefined>;
type NotifyLevel = "info" | "warning";

interface WarmRequestInit {
	method: "POST";
	headers: Record<string, string>;
	body: string;
	signal: AbortSignal;
}

interface WarmResponse {
	ok: boolean;
	status: number;
}

export interface WarmerDeps {
	setTimer(callback: () => void, delayMs: number): unknown;
	clearTimer(handle: unknown): void;
	fetchImpl(url: string, init: WarmRequestInit): Promise<WarmResponse>;
	resolveAuth(model: AnyModel): Promise<ResolvedAuth>;
	isIdle(): boolean;
	notify(message: string, level: NotifyLevel): void;
	/**
	 * Updates or clears the persistent cache warmer status.
	 * @param message - {string | undefined} The status message to show, or undefined to clear it.
	 */
	setStatus(message: string | undefined): void;
	env: EnvRecord;
}

export interface Warmer {
	capture(payload: unknown, model: AnyModel | undefined): void;
	updateModel(model: AnyModel | undefined): void;
	arm(): void;
	cancel(): void;
	dispose(): void;
	fireNow(): Promise<void>;
	readonly warmCount: number;
}

interface WarmerState {
	lastPayload: unknown;
	lastModel: AnyModel | undefined;
	timerHandle: unknown;
	inFlight: boolean;
	disposed: boolean;
	warmCount: number;
	lastWarmStatus: string;
	activeRequest: AbortController | undefined;
	targetVersion: number;
	activityGeneration: number;
	armRequested: boolean;
}

interface WarmTarget {
	model: AnyModel;
	payload: unknown;
	version: number;
	activityGeneration: number;
}

export function isDisabled(env: EnvRecord): boolean {
	return env[DISABLE_ENV_VAR] === "1";
}

export function resolveIntervalMs(env: EnvRecord): number {
	const raw = env[INTERVAL_ENV_VAR];
	const parsed = raw === undefined ? Number.NaN : Number(raw);
	const candidate = Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_INTERVAL_MS;
	return Math.min(Math.max(candidate, MIN_INTERVAL_MS), MAX_INTERVAL_MS);
}

interface ProviderDialect {
	endpoint(model: AnyModel): string;
	minimizeBody(payload: unknown): unknown;
	versionHeaders(): Record<string, string>;
	applyBareKey(headers: Record<string, string>, apiKey: string): void;
}

function stripTrailingSlash(baseUrl: unknown): string {
	return String(baseUrl).replace(/\/+$/, "");
}

function cloneMinimalBody(payload: unknown): Record<string, unknown> | undefined {
	if (!payload || typeof payload !== "object") return undefined;
	const clone: Record<string, unknown> = { ...(payload as Record<string, unknown>) };
	clone.stream = false;
	// JSON.stringify drops undefined-valued keys, so this excludes tool_choice from the wire payload.
	clone.tool_choice = undefined;
	return clone;
}

function minimizeAnthropicBody(payload: unknown): unknown {
	const clone = cloneMinimalBody(payload);
	if (!clone) return payload;
	clone.max_tokens = 1;
	// A minimized thinking block would reject max_tokens=1, so drop it entirely.
	clone.thinking = undefined;
	return clone;
}

function minimizeOpenAiCompletionsBody(payload: unknown): unknown {
	const clone = cloneMinimalBody(payload);
	if (!clone) return payload;
	const cap = "reasoning_effort" in clone ? REASONING_OUTPUT_FLOOR : 1;
	const outputTokenField = "max_completion_tokens" in clone ? "max_completion_tokens" : "max_tokens";
	clone[outputTokenField] = cap;
	return clone;
}

function minimizeOpenAiResponsesBody(payload: unknown): unknown {
	const clone = cloneMinimalBody(payload);
	if (!clone) return payload;
	const hasReasoning = typeof clone.reasoning === "object" && clone.reasoning !== null;
	clone.max_output_tokens = hasReasoning ? REASONING_OUTPUT_FLOOR : 1;
	clone.store = false;
	return clone;
}

function anthropicVersionHeaders(): Record<string, string> {
	return { "anthropic-version": ANTHROPIC_VERSION };
}

function noVersionHeaders(): Record<string, string> {
	return {};
}

function applyAnthropicBareKey(headers: Record<string, string>, apiKey: string): void {
	headers["x-api-key"] = apiKey;
}

function applyBearerBareKey(headers: Record<string, string>, apiKey: string): void {
	headers.authorization = `Bearer ${apiKey}`;
}

const DIALECTS: Partial<Record<string, ProviderDialect>> = {
	[ANTHROPIC_API]: {
		endpoint: (model) => `${stripTrailingSlash(model.baseUrl)}/v1/messages`,
		minimizeBody: minimizeAnthropicBody,
		versionHeaders: anthropicVersionHeaders,
		applyBareKey: applyAnthropicBareKey,
	},
	[OPENAI_COMPLETIONS_API]: {
		endpoint: (model) => `${stripTrailingSlash(model.baseUrl)}/chat/completions`,
		minimizeBody: minimizeOpenAiCompletionsBody,
		versionHeaders: noVersionHeaders,
		applyBareKey: applyBearerBareKey,
	},
	[OPENAI_RESPONSES_API]: {
		endpoint: (model) => `${stripTrailingSlash(model.baseUrl)}/responses`,
		minimizeBody: minimizeOpenAiResponsesBody,
		versionHeaders: noVersionHeaders,
		applyBareKey: applyBearerBareKey,
	},
};

export function getDialect(model: AnyModel | undefined): ProviderDialect | undefined {
	if (!model) return undefined;
	return DIALECTS[model.api];
}

export function isWarmableModel(model: AnyModel | undefined): model is AnyModel {
	return getDialect(model) !== undefined;
}

function hasAuthorizationHeader(headers: Record<string, string>): boolean {
	return Object.keys(headers).some((key) => key.toLowerCase() === "authorization");
}

function buildWarmHeaders(dialect: ProviderDialect, auth: AuthorizedAuth): Record<string, string> {
	const headers: Record<string, string> = {
		"content-type": "application/json",
		...dialect.versionHeaders(),
		...(auth.headers ?? {}),
	};
	if (!hasAuthorizationHeader(headers) && auth.apiKey) dialect.applyBareKey(headers, auth.apiKey);
	return headers;
}

function withTimeoutSignal(deps: WarmerDeps, delayMs: number): { controller: AbortController; dispose: () => void } {
	const controller = new AbortController();
	const handle = deps.setTimer(() => controller.abort(), delayMs);
	return { controller, dispose: () => deps.clearTimer(handle) };
}

function cancelTimer(deps: WarmerDeps, state: WarmerState): boolean {
	if (state.timerHandle === undefined) return false;
	deps.clearTimer(state.timerHandle);
	state.timerHandle = undefined;
	return true;
}

function cancelWarm(deps: WarmerDeps, state: WarmerState): void {
	state.activityGeneration += 1;
	state.armRequested = false;
	const cancelled = cancelTimer(deps, state);
	if (cancelled && !state.inFlight) deps.setStatus(state.lastWarmStatus || undefined);
}

function hasEligibleWarmTarget(deps: WarmerDeps, state: WarmerState): boolean {
	return !state.disposed && state.lastPayload !== undefined && isWarmableModel(state.lastModel) && !isDisabled(deps.env);
}

function canScheduleWarm(deps: WarmerDeps, state: WarmerState): boolean {
	return hasEligibleWarmTarget(deps, state) && !state.inFlight;
}

function currentWarmTarget(deps: WarmerDeps, state: WarmerState): WarmTarget | undefined {
	const model = state.lastModel;
	if (!canScheduleWarm(deps, state) || !deps.isIdle() || !isWarmableModel(model)) return undefined;
	return { model, payload: state.lastPayload, version: state.targetVersion, activityGeneration: state.activityGeneration };
}

function hasCurrentWarmTarget(state: WarmerState, target: WarmTarget): boolean {
	return !state.disposed && state.targetVersion === target.version;
}

function clearWarmStatus(deps: WarmerDeps, state: WarmerState): void {
	state.lastWarmStatus = "";
	deps.setStatus(undefined);
}

function publishWarmStatus(deps: WarmerDeps, state: WarmerState, message: string): void {
	if (state.disposed) return;
	state.lastWarmStatus = message;
	deps.setStatus(message);
}

function notifyWarmStatus(deps: WarmerDeps, state: WarmerState, message: string, level: NotifyLevel): void {
	if (state.disposed) return;
	state.lastWarmStatus = message;
	deps.notify(message, level);
}

function scheduleStatus(state: WarmerState): string {
	return state.lastWarmStatus ? `${state.lastWarmStatus}; next warm scheduled` : "next warm scheduled";
}

function armWarm(deps: WarmerDeps, state: WarmerState): void {
	if (state.disposed) return;
	cancelTimer(deps, state);
	if (!hasEligibleWarmTarget(deps, state)) {
		state.armRequested = false;
		clearWarmStatus(deps, state);
		return;
	}
	if (state.inFlight) {
		state.armRequested = true;
		return;
	}
	state.armRequested = false;
	state.timerHandle = deps.setTimer(() => {
		state.timerHandle = undefined;
		void fireWarmRequest(deps, state);
	}, resolveIntervalMs(deps.env));
	deps.setStatus(scheduleStatus(state));
}

function rearmWhenPossible(deps: WarmerDeps, state: WarmerState): void {
	armWarm(deps, state);
}

function publishSettledWarmStatus(deps: WarmerDeps, state: WarmerState): void {
	if (state.disposed) return;
	if (state.lastWarmStatus === "warming cache") {
		clearWarmStatus(deps, state);
		return;
	}
	deps.setStatus(state.lastWarmStatus || undefined);
}

async function sendWarmRequest(deps: WarmerDeps, state: WarmerState, target: WarmTarget): Promise<void> {
	const auth = await deps.resolveAuth(target.model);
	if (!hasCurrentWarmTarget(state, target)) return;
	if (!auth.ok) {
		notifyWarmStatus(deps, state, "warm request skipped", "warning");
		return;
	}
	const dialect = getDialect(target.model);
	if (!dialect) return;
	const timeout = withTimeoutSignal(deps, WARM_REQUEST_TIMEOUT_MS);
	state.activeRequest = timeout.controller;
	try {
		if (!hasCurrentWarmTarget(state, target)) return;
		const response = await deps.fetchImpl(dialect.endpoint(target.model), {
			method: "POST",
			headers: buildWarmHeaders(dialect, auth),
			body: JSON.stringify(dialect.minimizeBody(target.payload)),
			signal: timeout.controller.signal,
		});
		if (!hasCurrentWarmTarget(state, target)) return;
		if (!response.ok) {
			notifyWarmStatus(deps, state, `warm request failed (HTTP ${response.status})`, "warning");
			return;
		}
		state.warmCount += 1;
		notifyWarmStatus(deps, state, `warmed (count ${state.warmCount})`, "info");
	} finally {
		timeout.dispose();
		if (state.activeRequest === timeout.controller) state.activeRequest = undefined;
	}
}

async function fireWarmRequest(deps: WarmerDeps, state: WarmerState): Promise<void> {
	const target = currentWarmTarget(deps, state);
	if (!target) {
		rearmWhenPossible(deps, state);
		return;
	}
	state.inFlight = true;
	publishWarmStatus(deps, state, "warming cache");
	try {
		await sendWarmRequest(deps, state, target);
	} catch {
		if (hasCurrentWarmTarget(state, target)) notifyWarmStatus(deps, state, "warm request failed", "warning");
	} finally {
		state.inFlight = false;
	}
	const deferredArmRequested = state.armRequested;
	state.armRequested = false;
	if (!hasCurrentWarmTarget(state, target)) {
		if (deferredArmRequested && deps.isIdle()) rearmWhenPossible(deps, state);
		return;
	}
	if (deps.isIdle() && (deferredArmRequested || state.activityGeneration === target.activityGeneration)) {
		rearmWhenPossible(deps, state);
	} else {
		publishSettledWarmStatus(deps, state);
	}
}

/**
 * Creates a cache warmer for captured Anthropic provider requests.
 * @param deps - {WarmerDeps} The timer, request, state, and UI dependencies.
 * @returns {Warmer} A cache warmer controller.
 */
export function createWarmer(deps: WarmerDeps): Warmer {
	const state: WarmerState = {
		lastPayload: undefined,
		lastModel: undefined,
		timerHandle: undefined,
		inFlight: false,
		disposed: false,
		warmCount: 0,
		lastWarmStatus: "",
		activeRequest: undefined,
		targetVersion: 0,
		activityGeneration: 0,
		armRequested: false,
	};

	return {
		capture: (payload, model) => {
			if (state.inFlight) clearWarmStatus(deps, state);
			state.lastPayload = payload;
			state.lastModel = model;
			state.targetVersion += 1;
		},
		updateModel: (_model) => {
			if (state.disposed) return;
			state.lastPayload = undefined;
			state.lastModel = undefined;
			state.targetVersion += 1;
			state.armRequested = false;
			cancelTimer(deps, state);
			clearWarmStatus(deps, state);
		},
		arm: () => armWarm(deps, state),
		cancel: () => cancelWarm(deps, state),
		dispose: () => {
			state.disposed = true;
			state.targetVersion += 1;
			state.armRequested = false;
			cancelTimer(deps, state);
			clearWarmStatus(deps, state);
			state.activeRequest?.abort();
			state.activeRequest = undefined;
		},
		fireNow: () => fireWarmRequest(deps, state),
		get warmCount() {
			return state.warmCount;
		},
	};
}

function createRealDeps(getCtx: () => ExtensionContext | undefined): WarmerDeps {
	return {
		setTimer: (callback, delayMs) => {
			const handle = setTimeout(callback, delayMs) as unknown as { unref?: () => void };
			handle.unref?.();
			return handle;
		},
		clearTimer: (handle) => clearTimeout(handle as Parameters<typeof clearTimeout>[0]),
		fetchImpl: (url, init) => fetch(url, init),
		resolveAuth: (model) => {
			const ctx = getCtx();
			if (!ctx) return Promise.resolve({ ok: false, error: "pi-cache-warmer: extension context not ready yet" });
			return ctx.modelRegistry.getApiKeyAndHeaders(model);
		},
		isIdle: () => getCtx()?.isIdle() ?? false,
		notify: (message, level) => {
			const ctx = getCtx();
			if (ctx?.hasUI) ctx.ui.notify(`pi-cache-warmer: ${message}`, level);
		},
		setStatus: (message) => {
			const ctx = getCtx();
			if (ctx?.hasUI) ctx.ui.setStatus("pi-cache-warmer", message);
		},
		env: process.env,
	};
}

function guardHandler<E>(handler: (event: E, ctx: ExtensionContext) => void): (event: E, ctx: ExtensionContext) => void {
	return (event, ctx) => {
		try {
			handler(event, ctx);
		} catch {
			// extension handlers must never propagate into pi's event loop
		}
	};
}

export default function piCacheWarmer(pi: ExtensionAPI): void {
	const guard = pi as unknown as Record<PropertyKey, unknown>;
	if (guard[INSTALL_SYMBOL]) return;
	guard[INSTALL_SYMBOL] = true;

	let currentCtx: ExtensionContext | undefined;
	const warmer = createWarmer(createRealDeps(() => currentCtx));

	pi.on("before_provider_request", (event, ctx) => {
		currentCtx = ctx;
		try {
			warmer.capture(event.payload, ctx.model);
			warmer.cancel();
		} catch {
			// never let a capture failure block the real provider request
		}
		return event.payload;
	});

	pi.on(
		"model_select",
		guardHandler((event, ctx) => {
			currentCtx = ctx;
			warmer.updateModel(event.model);
		}),
	);

	pi.on(
		"agent_start",
		guardHandler((_event, ctx) => {
			currentCtx = ctx;
			warmer.cancel();
		}),
	);

	pi.on(
		"input",
		guardHandler((_event, ctx) => {
			currentCtx = ctx;
			warmer.cancel();
		}),
	);

	pi.on(
		"agent_end",
		guardHandler((_event, ctx) => {
			currentCtx = ctx;
			warmer.arm();
		}),
	);

	pi.on(
		"session_shutdown",
		guardHandler(() => {
			warmer.dispose();
		}),
	);
}

import type { ExtensionAPI, ExtensionContext } from "@mariozechner/pi-coding-agent";

const INSTALL_SYMBOL = Symbol.for("klock.pi-cache-warmer.installed");

const ANTHROPIC_API = "anthropic-messages";
const ANTHROPIC_VERSION = "2023-06-01";
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
}

interface WarmTarget {
	model: AnyModel;
	payload: unknown;
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

export function isAnthropicModel(model: AnyModel | undefined): model is AnyModel {
	return model?.api === ANTHROPIC_API;
}

export function anthropicEndpoint(model: AnyModel): string {
	return `${String(model.baseUrl).replace(/\/+$/, "")}/v1/messages`;
}

export function buildWarmBody(payload: unknown): unknown {
	if (!payload || typeof payload !== "object") return payload;
	const clone: Record<string, unknown> = { ...(payload as Record<string, unknown>) };
	clone.max_tokens = 1;
	clone.stream = false;
	// JSON.stringify drops undefined-valued keys, so this excludes tool_choice from the wire payload.
	clone.tool_choice = undefined;
	return clone;
}

function hasAuthorizationHeader(headers: Record<string, string>): boolean {
	return Object.keys(headers).some((key) => key.toLowerCase() === "authorization");
}

function buildWarmHeaders(auth: AuthorizedAuth): Record<string, string> {
	const headers: Record<string, string> = {
		"content-type": "application/json",
		"anthropic-version": ANTHROPIC_VERSION,
		...(auth.headers ?? {}),
	};
	if (!hasAuthorizationHeader(headers) && auth.apiKey) headers["x-api-key"] = auth.apiKey;
	return headers;
}

function withTimeoutSignal(deps: WarmerDeps, delayMs: number): { signal: AbortSignal; dispose: () => void } {
	const controller = new AbortController();
	const handle = deps.setTimer(() => controller.abort(), delayMs);
	return { signal: controller.signal, dispose: () => deps.clearTimer(handle) };
}

export function createWarmer(deps: WarmerDeps): Warmer {
	const state: WarmerState = {
		lastPayload: undefined,
		lastModel: undefined,
		timerHandle: undefined,
		inFlight: false,
		disposed: false,
		warmCount: 0,
	};

	function cancel(): void {
		if (state.timerHandle === undefined) return;
		deps.clearTimer(state.timerHandle);
		state.timerHandle = undefined;
	}

	function capture(payload: unknown, model: AnyModel | undefined): void {
		state.lastPayload = payload;
		state.lastModel = model;
	}

	function updateModel(model: AnyModel | undefined): void {
		state.lastModel = model;
	}

	function currentWarmTarget(): WarmTarget | undefined {
		if (state.disposed || state.inFlight) return undefined;
		if (!deps.isIdle()) return undefined;
		if (state.lastPayload === undefined) return undefined;
		if (!isAnthropicModel(state.lastModel)) return undefined;
		if (isDisabled(deps.env)) return undefined;
		return { model: state.lastModel, payload: state.lastPayload };
	}

	async function sendWarmRequest(model: AnyModel, payload: unknown): Promise<void> {
		const auth = await deps.resolveAuth(model);
		if (!auth.ok) {
			deps.notify(`warm request skipped: ${auth.error}`, "warning");
			return;
		}
		const timeout = withTimeoutSignal(deps, WARM_REQUEST_TIMEOUT_MS);
		try {
			await deps.fetchImpl(anthropicEndpoint(model), {
				method: "POST",
				headers: buildWarmHeaders(auth),
				body: JSON.stringify(buildWarmBody(payload)),
				signal: timeout.signal,
			});
			state.warmCount += 1;
			deps.notify(`warmed (count ${state.warmCount})`, "info");
		} finally {
			timeout.dispose();
		}
	}

	async function fireNow(): Promise<void> {
		const target = currentWarmTarget();
		if (!target) return;
		state.inFlight = true;
		try {
			await sendWarmRequest(target.model, target.payload);
		} catch (error) {
			deps.notify(`warm request failed: ${String(error)}`, "warning");
		} finally {
			state.inFlight = false;
		}
		if (!state.disposed && deps.isIdle()) arm();
	}

	function arm(): void {
		if (state.disposed) return;
		cancel();
		state.timerHandle = deps.setTimer(() => {
			state.timerHandle = undefined;
			void fireNow();
		}, resolveIntervalMs(deps.env));
	}

	function dispose(): void {
		state.disposed = true;
		cancel();
	}

	return {
		capture,
		updateModel,
		arm,
		cancel,
		dispose,
		fireNow,
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

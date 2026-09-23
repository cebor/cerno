import { Answers } from "./answers.ts";
import { SystemOneBuilder } from "./builder.ts";
import { ApiError, TransportError, UnexpectedResponse } from "./errors.ts";
import type {
  ErrorResponse,
  ModelsResponse,
  SystemOneRequest,
  SystemOneResponse,
} from "./types.ts";

export interface ClientOptions {
  /** Defaults to `http://localhost:3000`. */
  baseUrl?: string;
  /** Milliseconds before a request is aborted. Defaults to 60000. */
  timeoutMs?: number;
  /** Sent with every request. */
  headers?: Record<string, string>;
  /** Injected for testing, or to route through a custom transport. */
  fetch?: typeof globalThis.fetch;
}

const DEFAULT_TIMEOUT_MS = 60_000;

/**
 * Client for the cerno service.
 *
 * ```ts
 * const client = new Client({ baseUrl: "http://localhost:3000" });
 * const answers = await client
 *   .systemone("Ticket: server room at 31C, rising.")
 *   .noul("urgent", "Is this urgent?")
 *   .choice("team", "Which team?", ["IT", "Facility", "HR"])
 *   .send();
 * ```
 */
export class Client {
  private readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly headers: Record<string, string>;
  private readonly doFetch: typeof globalThis.fetch;

  constructor(options: ClientOptions | string = {}) {
    const opts = typeof options === "string" ? { baseUrl: options } : options;

    this.baseUrl = (opts.baseUrl ?? "http://localhost:3000").replace(/\/+$/, "");
    this.timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.headers = opts.headers ?? {};
    this.doFetch = opts.fetch ?? globalThis.fetch;
  }

  /** Start a request about `state`. */
  systemone(state: string): SystemOneBuilder {
    return new SystemOneBuilder(state, async (body) => {
      const response = await this.request<SystemOneResponse>("POST", "/v1/systemone", body);
      return new Answers(response);
    });
  }

  /** The models this service will answer for. */
  models(): Promise<ModelsResponse> {
    return this.request<ModelsResponse>("GET", "/v1/models");
  }

  async health(): Promise<boolean> {
    try {
      const response = await this.doFetch(`${this.baseUrl}/health`);
      return response.ok;
    } catch {
      return false;
    }
  }

  private async request<T>(method: string, path: string, body?: unknown): Promise<T> {
    // An explicit abort, so a wedged host surfaces as a timeout rather than hanging forever.
    const abort = AbortSignal.timeout(this.timeoutMs);

    let response: Response;
    let text: string;
    try {
      response = await this.doFetch(`${this.baseUrl}${path}`, {
        method,
        signal: abort,
        headers: {
          ...(body === undefined ? {} : { "content-type": "application/json" }),
          ...this.headers,
        },
        body: body === undefined ? undefined : JSON.stringify(body),
      });
      text = await response.text();
    } catch (err) {
      throw new TransportError(`could not reach cerno: ${String(err)}`, err);
    }

    if (response.ok) {
      try {
        return JSON.parse(text) as T;
      } catch {
        throw new UnexpectedResponse(response.status, text);
      }
    }

    let parsed: ErrorResponse | undefined;
    try {
      const candidate = JSON.parse(text);
      if (candidate && typeof candidate.code === "string") parsed = candidate;
    } catch {
      // Falls through to UnexpectedResponse below.
    }

    // Not our error shape, so do not pretend to know what went wrong.
    if (!parsed) throw new UnexpectedResponse(response.status, text);
    throw new ApiError(response.status, parsed);
  }
}

export type { SystemOneRequest };

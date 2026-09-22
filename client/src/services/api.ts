/** API boundary - no persistence, credentials, or user data is fabricated in the browser. */

import type {
  ConversationSummary,
  Message,
  MemoryRecord,
  AuthUser,
  TokenResponse,
  AnalyticsRead,
  AnalyticsOverview,
  AnalyticsPoint,
  AnalyticsItem,
  ProfileUpdate,
  UserSettings,
} from "@/lib/types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "/api/v1";
const TOKEN_STORAGE_KEY = "ai-memory-hub.access-token";
const LOCAL_CONNECTOR_URL = import.meta.env.VITE_LOCAL_CONNECTOR_URL ?? "http://localhost:8765";

let accessToken = sessionStorage.getItem(TOKEN_STORAGE_KEY);

export class ApiUnavailableError extends Error {
  constructor(message = "AI Memory Hub backend is not connected.") {
    super(message);
    this.name = "ApiUnavailableError";
  }
}

export function setAccessToken(token: string | null) {
  accessToken = token;
  if (token) sessionStorage.setItem(TOKEN_STORAGE_KEY, token);
  else sessionStorage.removeItem(TOKEN_STORAGE_KEY);
}

export function getAccessToken() {
  return accessToken;
}

async function localFetch(input: string, init: RequestInit = {}) {
  // Chrome 142+ gates public-site -> localhost requests behind Local Network Access.
  // Declaring the target address space lets the browser apply the intended local-network policy.
  return fetch(input, {
    ...init,
    ...({ targetAddressSpace: "loopback" } as RequestInit & { targetAddressSpace: "loopback" }),
  });
}

async function localRequest<T>(path: string): Promise<T> {
  try {
    const response = await localFetch(`${LOCAL_CONNECTOR_URL}${path}`);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || "Local AI Connector is unavailable.");
    return payload as T;
  } catch (error) {
    if (error instanceof TypeError) {
      throw new Error("Browser could not access the local Ollama connector. Allow Local Network Access for ai-memory-hub-phi.vercel.app in Chrome.");
    }
    throw error;
  }
}

/** The Google start route must be opened as a document navigation, not fetched. */
export function getGoogleOAuthStartUrl() {
  return `${API_BASE_URL}/auth/google/start`;
}

export interface GenerateRequest {
  message: string;
  provider: string;
  model_key: string;
}

export interface ProviderRead {
  id: string;
  provider: string;
  is_enabled: boolean;
  created_at: string;
}

export interface ProviderModelRead {
  model_key: string;
  display_name: string;
  context_length?: number;
  is_local: boolean;
}

export interface ModelRead {
  id: string;
  provider: string;
  model_key: string;
  display_name: string;
  is_local: boolean;
  is_active: boolean;
}

interface ConversationResponse {
  id: string;
  title: string;
  selected_model_id: string | null;
  is_archived: boolean;
  created_at: string;
}

interface MemoryResponse {
  id: string;
  content: string;
  category: string;
  source_conversation_id: string | null;
  importance: number;
  confidence: number;
  is_archived: boolean;
  is_pinned: boolean;
  created_at: string;
}

interface MemorySearchResponse extends MemoryResponse {
  score: number;
}

export interface MemoryCreateRequest {
  content: string;
  category?: string;
  source_conversation_id?: string | null;
  importance?: number;
  confidence?: number;
}

export interface MemoryUpdateRequest {
  content?: string;
  category?: string;
  importance?: number;
  confidence?: number;
  is_archived?: boolean;
  is_pinned?: boolean;
}

function toConversationSummary(item: ConversationResponse): ConversationSummary {
  return {
    id: item.id,
    title: item.title,
    updatedAt: item.created_at,
    model: item.selected_model_id ?? "No model selected",
    memoryUsed: false,
    selected_model_id: item.selected_model_id,
  };
}

function toMemoryRecord(item: MemoryResponse): MemoryRecord {
  return {
    id: item.id,
    title: item.content.slice(0, 56),
    content: item.content,
    category: item.category,
    source: item.source_conversation_id ?? "Manual entry",
    createdAt: item.created_at,
    updatedAt: item.created_at,
    pinned: item.is_pinned,
    status: item.is_archived ? "archived" : "active",
  };
}

function handleUnauthorized() {
  setAccessToken(null);
  window.dispatchEvent(new Event("ai-memory-hub.auth-expired"));
}

async function request<T>(path: string, init: RequestInit = {}, responseType: "json" | "blob" = "json"): Promise<T> {
  try {
    const headers = new Headers(init.headers);
    headers.set("Content-Type", "application/json");
    if (accessToken) headers.set("Authorization", `Bearer ${accessToken}`);
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers,
      credentials: "include",
    });

    if (!response.ok) {
      if (response.status === 401) handleUnauthorized();
      const text = await response.text();
      let msg = `Request failed (${response.status})`;
      try {
        const payload = JSON.parse(text);
        if (payload.detail) msg = payload.detail;
      } catch {}
      throw new Error(msg);
    }

    if (response.status === 204) return null as T;
    if (responseType === "blob") return (await response.blob()) as T;
    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof Error) throw error;
    throw new ApiUnavailableError();
  }
}

export const api = {
  auth: {
    signIn: (email: string, password: string) =>
      request<TokenResponse>("/auth/login", { method: "POST", body: JSON.stringify({ email, password }) }),
    register: (email: string, password: string, fullName: string) =>
      request<{ detail: string }>("/auth/register", { method: "POST", body: JSON.stringify({ email, password, full_name: fullName || null }) }),
    verifyEmailOtp: (email: string, otp: string) =>
      request<TokenResponse>("/auth/verify-email", { method: "POST", body: JSON.stringify({ email, otp }) }),
    resendVerification: (email: string) =>
      request<{ detail: string }>("/auth/resend-verification", { method: "POST", body: JSON.stringify({ email }) }),
    requestPasswordReset: (email: string) =>
      request<{ detail: string }>("/auth/forgot-password", { method: "POST", body: JSON.stringify({ email }) }),
    verifyReset: (email: string, otp: string) =>
      request<{ reset_token: string }>("/auth/verify-reset", { method: "POST", body: JSON.stringify({ email, otp }) }),
    resetPassword: (resetToken: string, newPassword: string) =>
      request<void>("/auth/reset-password", { method: "POST", body: JSON.stringify({ reset_token: resetToken, new_password: newPassword }) }),
    me: () => request<AuthUser>("/auth/me"),
    logout: () => request<void>("/auth/logout", { method: "POST" }),
  },
  models: {
    listRegistry: () => request<ModelRead[]>("/models"),
  },
  providers: {
    listConfigured: () => request<ProviderRead[]>("/providers"),
    configure: (provider: string, apiKey: string | null, isEnabled: boolean) =>
      request<ProviderRead>("/providers", {
        method: "POST",
        body: JSON.stringify({ provider, api_key: apiKey, is_enabled: isEnabled }),
      }),
    createCredentials: (provider: string, apiKey: string | null, isEnabled = true) =>
      request<ProviderRead>(`/providers/${provider}/credentials`, { method: "POST", body: JSON.stringify({ api_key: apiKey, is_enabled: isEnabled }) }),
    update: (provider: string, patch: { api_key?: string | null; is_enabled?: boolean }) =>
      request<ProviderRead>(`/providers/${provider}`, { method: "PUT", body: JSON.stringify(patch) }),
    remove: (provider: string) => request<void>(`/providers/${provider}`, { method: "DELETE" }),
    removeCredentials: (provider: string) => request<void>(`/providers/${provider}/credentials`, { method: "DELETE" }),
    test: (provider: string) => request<{ provider: string; status: string }>(`/providers/${provider}/test`, { method: "POST" }),
    listModels: (provider: string) => request<ProviderModelRead[]>(`/providers/${provider}/models`),
    registerLocalModels: (models: Array<{ model_key: string; display_name?: string }>) =>
      request<ModelRead[]>("/providers/ollama/local-models", { method: "POST", body: JSON.stringify(models) }),
  },
  conversations: {
    list: async () => (await request<ConversationResponse[]>("/conversations")).map(toConversationSummary),
    create: async (title?: string) => toConversationSummary(await request<ConversationResponse>("/conversations", { method: "POST", body: JSON.stringify({ title }) })),
    update: (conversationId: string, patch: { title?: string; selected_model_id?: string | null }) =>
      request<ConversationResponse>(`/conversations/${conversationId}`, { method: "PATCH", body: JSON.stringify(patch) }).then(toConversationSummary),
    remove: (conversationId: string) => request<void>(`/conversations/${conversationId}`, { method: "DELETE" }),
    listMessages: (conversationId: string) => request<Message[]>(`/conversations/${conversationId}/messages`),
    export: () => request<Blob>("/privacy/export/conversations", {}, "blob"),
    prepareLocalGeneration: (conversationId: string, message: string, modelKey: string) =>
      request<{ message_id: string; model_key: string; messages: Array<{ role: string; content: string }> }>(
        `/conversations/${conversationId}/prepare-local-generation`,
        { method: "POST", body: JSON.stringify({ message, model_key: modelKey }) }
      ),
    completeLocalGeneration: (conversationId: string, userMessageId: string, content: string) =>
      request<{ message_id: string }>(
        `/conversations/${conversationId}/complete-local-generation`,
        { method: "POST", body: JSON.stringify({ user_message_id: userMessageId, content }) }
      ),
    generate: (
      conversationId: string,
      req: GenerateRequest,
      onChunk: (text: string) => void,
      onDone: (messageId: string) => void,
      onError: (err: Error) => void,
      signal?: AbortSignal
    ): void => {
      fetch(`${API_BASE_URL}/conversations/${conversationId}/generate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
        },
        body: JSON.stringify(req),
        credentials: "include",
        signal,
      })
        .then(async (response) => {
          if (!response.ok) {
            if (response.status === 401) handleUnauthorized();
            const text = await response.text();
            let msg = `Request failed (${response.status})`;
            try {
              const payload = JSON.parse(text);
              if (payload.detail) msg = payload.detail;
            } catch {}
            throw new Error(msg);
          }
          const reader = response.body?.getReader();
          if (!reader) throw new Error("Response body is not readable.");
          const decoder = new TextDecoder();
          let buffer = "";
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() ?? "";
            for (const line of lines) {
              if (!line.trim()) continue;
              try {
                const event = JSON.parse(line);
                if (event.type === "chunk") onChunk(event.text);
                else if (event.type === "done") onDone(event.message_id);
                else if (event.type === "error") {
                  const details: Record<string, string> = {
                    PROVIDER_AUTH_ERROR: "Provider authentication failed.",
                    PROVIDER_BILLING_OR_CREDITS: "Provider account or credits cannot currently serve this request.",
                    PROVIDER_FORBIDDEN: "Provider access was denied for this request.",
                    MODEL_NOT_FOUND: "Selected model is unavailable.",
                    PROVIDER_TIMEOUT: "Provider request timed out.",
                    PROVIDER_CONFLICT: "Provider could not accept this request in its current state.",
                    RATE_LIMITED: "Provider rate limit reached.",
                    PROVIDER_UNAVAILABLE: "Provider temporarily unavailable.",
                    CONTEXT_LIMIT_EXCEEDED: "Context limit exceeded.",
                    GENERATION_LIMIT_REACHED: "Generation limit reached.",
                    PROVIDER_ERROR: "The provider could not complete this request.",
                  };
                  const error = new Error(event.detail || details[event.code] || "Error during generation");
                  (error as Error & { code?: string }).code = event.code;
                  throw error;
                }
              } catch (error) {
                onError(error instanceof Error ? error : new Error(String(error)));
                return;
              }
            }
          }
        })
        .catch((error) => {
          if (error.name === "AbortError") return;
          onError(error instanceof Error ? error : new Error(String(error)));
        });
    },
  },
  memories: {
    list: async () => (await request<MemoryResponse[]>("/memories")).map(toMemoryRecord),
    create: (payload: MemoryCreateRequest) =>
      request<MemoryResponse>("/memories", { method: "POST", body: JSON.stringify(payload) }).then(toMemoryRecord),
    update: (memoryId: string, patch: MemoryUpdateRequest) =>
      request<MemoryResponse>(`/memories/${memoryId}`, { method: "PATCH", body: JSON.stringify(patch) }).then(toMemoryRecord),
    remove: (memoryId: string) => request<void>(`/memories/${memoryId}`, { method: "DELETE" }),
    archive: (memoryId: string) => request<MemoryResponse>(`/memories/${memoryId}/archive`, { method: "POST" }).then(toMemoryRecord),
    restore: (memoryId: string) => request<MemoryResponse>(`/memories/${memoryId}/restore`, { method: "POST" }).then(toMemoryRecord),
    pin: (memoryId: string) => request<MemoryResponse>(`/memories/${memoryId}/pin`, { method: "POST" }).then(toMemoryRecord),
    search: async (query: string, limit = 8) => {
      const boundedLimit = Math.min(Math.max(limit, 1), 20);
      return (await request<MemorySearchResponse[]>(`/memories/search?query=${encodeURIComponent(query)}&limit=${boundedLimit}`)).map(toMemoryRecord);
    },
    export: () => request<Blob>("/privacy/export/memories", {}, "blob"),
  },
  profile: {
    get: () => request<AuthUser>("/users/me"),
    update: (patch: ProfileUpdate) => request<AuthUser>("/users/me", { method: "PATCH", body: JSON.stringify(patch) }),
  },
  settings: {
    get: () => request<UserSettings>("/settings"),
    update: (patch: Partial<UserSettings>) => request<UserSettings>("/settings", { method: "PATCH", body: JSON.stringify(patch) }),
  },
  analytics: {
    get: () => request<AnalyticsRead>("/analytics"),
    overview: (days: number | null) => request<AnalyticsOverview>(`/analytics/overview${days ? `?days=${days}` : ""}`),
    activity: (days: number | null) => request<AnalyticsPoint[]>(`/analytics/activity${days ? `?days=${days}` : ""}`),
    memories: (days: number | null) => request<AnalyticsItem[]>(`/analytics/memories${days ? `?days=${days}` : ""}`),
    models: (days: number | null) => request<AnalyticsItem[]>(`/analytics/models${days ? `?days=${days}` : ""}`),
    providers: (days: number | null) => request<AnalyticsItem[]>(`/analytics/providers${days ? `?days=${days}` : ""}`),
  },
  privacy: {
    export: () => request<Record<string, unknown>>("/privacy/export"),
  },
  localConnector: {
    health: () => localRequest<{ status: string; detail?: string }>("/health"),
    status: () => localRequest<{ connected: boolean; status: string }>("/status"),
    models: () => localRequest<{ models: Array<{ name: string; size?: number; modified_at?: string }> }>("/models"),
    chat: async (
      model: string,
      messages: Array<{ role: string; content: string }>,
      onChunk: (text: string) => void,
      signal?: AbortSignal,
    ) => {
      let response: Response;
      try {
        response = await localFetch(`${LOCAL_CONNECTOR_URL}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model, messages, stream: true }),
          signal,
        });
      } catch (error) {
        if (error instanceof TypeError) {
          throw new Error("Browser could not access the local Ollama connector. Allow Local Network Access for ai-memory-hub-phi.vercel.app in Chrome.");
        }
        throw error;
      }
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || `Local Ollama request failed (${response.status})`);
      }
      const reader = response.body?.getReader();
      if (!reader) throw new Error("Local Ollama response body is not readable.");
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.trim()) continue;
          const event = JSON.parse(line);
          const text = event?.message?.content;
          if (typeof text === "string" && text) onChunk(text);
          if (event?.error) throw new Error(String(event.error));
        }
      }
      if (buffer.trim()) {
        const event = JSON.parse(buffer);
        const text = event?.message?.content;
        if (typeof text === "string" && text) onChunk(text);
        if (event?.error) throw new Error(String(event.error));
      }
    },
  },
};

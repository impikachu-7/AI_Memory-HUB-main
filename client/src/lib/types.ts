/** AI Memory Hub interface types — Quiet Intelligence Console: explicit ownership and user-scoped boundaries. */

export type ProviderId =
  | "google"
  | "openai"
  | "anthropic"
  | "ollama"
  | "deepseek"
  | "groq"
  | "openrouter";

export interface AnalyticsRead {
  conversations: number;
  messages: number;
  memories: number;
}

export interface AnalyticsOverview {
  total_conversations: number;
  total_messages: number;
  total_memories: number;
  total_ai_requests: number;
  successful_requests: number;
  failed_requests: number;
}

export interface AnalyticsPoint {
  date: string;
  conversations: number;
  messages: number;
  memories: number;
  successful_requests: number;
  failed_requests: number;
}

export interface AnalyticsItem {
  name: string;
  count: number;
  provider?: string | null;
  is_local?: boolean | null;
}

export interface UserSettings {
  memory_enabled: boolean;
  memory_retrieval_mode: "automatic" | "off";
  notifications_enabled: boolean;
}

export interface ProfileUpdate {
  full_name: string | null;
}

export type MemoryStatus = "active" | "archived";

export interface ProviderSummary {
  id: ProviderId;
  name: string;
  connected: boolean;
  local?: boolean;
  modelCount: number;
}

export interface AvailableModel {
  id: string;
  providerId: ProviderId;
  providerName: string;
  name: string;
  context: string;
  local?: boolean;
}

export interface MemoryRecord {
  id: string;
  title: string;
  content: string;
  category: string;
  source: string;
  createdAt: string;
  updatedAt: string;
  pinned?: boolean;
  status: MemoryStatus;
}

export interface ConversationSummary {
  id: string;
  title: string;
  updatedAt: string;
  model: string;
  memoryUsed: boolean;
  selected_model_id?: string | null;
}

export interface Message {
  id: string;
  conversation_id: string;
  role: "user" | "assistant" | "system";
  content: string;
  provider?: string | null;
  model_id: string | null;
  created_at: string;
}

export interface ModelRead {
  id: string;
  provider: string;
  model_key: string;
  display_name: string;
  is_local: boolean;
  is_active: boolean;
  context_window?: number | null;
  max_output_tokens?: number | null;
  supports_streaming?: boolean | null;
  supports_temperature?: boolean | null;
}

export interface ProviderRead {
  id: string;
  provider: string;
  is_enabled: boolean;
  created_at: string;
}

export interface ApiErrorPayload {
  message: string;
  code?: string;
  status?: number;
}

export interface AuthUser {
  id: string;
  email: string;
  full_name: string | null;
  is_email_verified: boolean;
}

export interface TokenResponse {
  access_token: string;
  token_type: "bearer";
  user: AuthUser | null;
}

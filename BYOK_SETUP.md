# BYOK Setup

AI Memory Hub uses Bring Your Own Key. Each user adds provider credentials from the authenticated provider settings flow. No shared provider API keys are required.

Users provide their own OpenAI, Gemini, Anthropic, DeepSeek, Groq, OpenRouter, KIE, and NVIDIA NIM keys in Provider Settings. No shared KIE key is required in Render. Groq model availability and limits depend on its current developer tier. OpenRouter free models are identified from current model pricing metadata. NVIDIA NIM exposes available models through its API catalog; free-endpoint status is shown only when provider metadata supplies it.

Required application secrets include:

- `DATABASE_URL`
- `JWT_SECRET`
- `ENCRYPTION_KEY` (the legacy `PROVIDER_ENCRYPTION_KEY` name is also accepted)
- `FRONTEND_ORIGINS`

Credentials are encrypted before storage, scoped by `user_id`, and never returned in provider responses or exports. The backend decrypts a credential only while constructing a provider request. The UI receives connection state, never the raw key.

Ollama runs locally and does not require an API key. Do not put cloud keys in the connector environment.

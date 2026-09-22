# BYOK Setup

AI Memory Hub uses Bring Your Own Key. Each user adds provider credentials from the authenticated provider settings flow. The server does not require shared OpenAI, Groq, Gemini, Anthropic, or OpenRouter keys.

Required application secrets include:

- `DATABASE_URL`
- `JWT_SECRET`
- `ENCRYPTION_KEY` (the legacy `PROVIDER_ENCRYPTION_KEY` name is also accepted)
- `FRONTEND_ORIGINS`

Credentials are encrypted before storage, scoped by `user_id`, and never returned in provider responses or exports. The backend decrypts a credential only while constructing a provider request. The UI receives connection state, never the raw key.

Ollama is local and does not need a cloud key. Do not put cloud keys in the connector environment.

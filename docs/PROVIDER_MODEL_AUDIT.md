# Provider and model audit

Reviewed 2026-10-29. This audit is about repository behavior and public provider policy; no customer API credentials were used and no live inference was run.

## Provider status

| Provider | User API key in settings | Discovery | Current availability | Generation/streaming | Free-only selection |
| --- | --- | --- | --- | --- | --- |
| OpenAI | BYOK; encrypted | Official model list | NOT TESTED (no user key) | Mocked tests only | No models: public API model pricing is paid |
| Google Gemini | BYOK; encrypted | Official model list, filtered to supported text generation IDs | NOT TESTED (no user key) | Mocked tests only | Curated free-tier IDs, intersected with the live account catalog |
| Anthropic | BYOK; encrypted | Official model list | NOT TESTED (no user key) | Mocked tests only | No models: API billing is usage-based |
| DeepSeek | BYOK; encrypted | Official model list | NOT TESTED (no user key) | Mocked tests only | No models: API prices are nonzero |
| Groq | BYOK; encrypted | Official model list intersected with the documented free-plan chat allowlist | NOT TESTED (no user key) | Mocked tests only | Free-plan chat IDs only; quotas/rate limits still apply |
| OpenRouter | BYOK; encrypted | Official model list and current text-chat pricing metadata | NOT TESTED (no user key) | Mocked tests only | Only models with numeric zero prompt AND completion pricing; `openrouter/free` is offered only when a free chat model is discovered |
| KIE | BYOK in Provider Settings; encrypted | Official supported model list | NOT TESTED (no user key) | Mocked tests only | No models: current model pricing is credit-based |
| NVIDIA NIM | BYOK; encrypted | NVIDIA OpenAI-compatible `/v1/models`; optional metadata handled without requiring it | NOT TESTED (no user key) | Mocked tests only | Only rows explicitly marked Free Endpoint/free or with zero prompt+completion pricing |
| Ollama | No cloud key | Installed local models through the browser connector | Local integration tests only; no user machine inspected | Mocked tests only | Installed local models only; requests stay browser → connector → Ollama |

## Selectable catalog

The exact dynamic model counts depend on each user's credentials, account entitlements, live catalog and locally installed Ollama models. Those counts cannot be truthfully stated without those inputs. The currently documented Gemini allowlist contains six model IDs, and the Groq free-plan chat allowlist contains three IDs; both are still intersected with the provider's live model list. OpenRouter and NVIDIA are discovered dynamically and require fresh positive free evidence. Paid or unknown-price models from OpenAI, Anthropic, DeepSeek and KIE are excluded; their provider integrations and encrypted credentials remain available for future policy changes.

The backend returns only active free-verified models for selection, deactivates removed/paid/unknown rows after a successful refresh, and retains historical model rows/messages. Discovery failures do not delete cached registry records or certify them as current. Generation checks the user's refreshed/cached catalog before saving the user message or calling inference. No migration is needed; the repository already has nullable `model_registry.is_free` and provider/model uniqueness.

## Root causes fixed

1. NVIDIA discovery treated optional endpoint/modality metadata as mandatory. Valid chat model entries with canonical IDs but only normal `/models` fields were discarded. The provider now accepts unknown capability metadata unless the ID is a known non-chat task or metadata explicitly rules out text chat. Free status remains a separate strict gate.
2. The old registry was treated as evidence that a model remained available. A stale or seed row could reach generation, and older catalog entries could remain active after successful refresh. Selection now requires the exact model key in the user's verified catalog; successful refreshes deactivate disappeared rows without deleting history.
3. The frontend canceled the NDJSON stream on `done`, then replaced the chat state with a history fetch. That could cut off backend post-response work and cause a completed assistant message to disappear if the refresh raced or failed. It now drains the stream, renders the persisted assistant response immediately, and ignores a late history refresh for a conversation that is no longer active.
4. Unbounded SDK retries and long shared waits could make failures look like a stuck chat. Cloud provider SDK clients now use bounded timeouts and no hidden retries. The generation endpoint records request ID, first-token time, chunk count, total duration, retry count and safe error category without logging prompt or key material. OpenRouter may retry a free route only for selected transient/unavailable statuses before any output; HTTP 402 never triggers a fallback.

## Model identity and security

Provider discovery stores the provider's exact model ID. The same ID is checked by the server, sent to the provider and stored in message metadata. No shared NVIDIA key or Render provider key is required. User keys continue through the existing encrypted credential path; tests use mock credentials and mock provider responses only.

## Verification status

Automated provider discovery and chat-generation tests use mocks and do not spend credits. Live generation for every provider/model remains NOT TESTED without user BYOK credentials. Free-tier availability can also depend on geography, account eligibility, quotas and current provider terms. Provider Settings connection/model refresh remains the user's live check; an API model list cannot prove a successful inference call.


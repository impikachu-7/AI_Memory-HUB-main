# API Documentation

The authenticated API is mounted at `/api/v1` and accepts bearer tokens.

## Core routes

- `POST /auth/register`, `POST /auth/verify-email`, `POST /auth/login`, `POST /auth/logout`
- `GET /conversations`, `POST /conversations`, `PATCH /conversations/{id}`, `DELETE /conversations/{id}`
- `GET /conversations/{id}/messages`, `POST /conversations/{id}/messages`
- `POST /conversations/{id}/generate` streams newline-delimited JSON events: `chunk`, `done`, or `error`.
- `GET /memories`, `POST /memories`, `PATCH /memories/{id}`, `DELETE /memories/{id}`
- `POST /memories/{id}/archive`, `/restore`, and `/pin`; `GET /memories/search`
- `GET /providers`, `POST /providers`, `PUT /providers/{provider}`, `DELETE /providers/{provider}`
- `GET /providers/{provider}/models` and `GET /models`
- `GET /settings`, `PATCH /settings`
- `GET /analytics`, `GET /analytics/overview`, `GET /analytics/activity`, `GET /analytics/memories`, `GET /analytics/models`, `GET /analytics/providers`
- `GET /privacy/export`

Analytics endpoints accept an optional `days` query parameter (`7`, `30`, `90`, or an all-time request without the parameter) and return only authenticated-user data.

Provider credentials use `POST /providers/{provider}/credentials`, `GET /providers`, `POST /providers/{provider}/test`, and `DELETE /providers/{provider}/credentials`. Keys are never returned.

All resource queries are scoped to the authenticated user. Provider keys are never returned.

## Local connector

The separate connector runs at `http://127.0.0.1:8765` and exposes `/health`, `/status`, `/models`, `/chat`, and `/generate`. It forwards only local Ollama requests.

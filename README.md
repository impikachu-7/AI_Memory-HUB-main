# AI Memory Hub

AI Memory Hub is a user-owned memory layer for cloud and local AI models. React/Vite provides the workspace UI; FastAPI, SQLAlchemy, PostgreSQL, and ChromaDB provide authenticated storage and retrieval.

## Architecture

- Cloud providers use each authenticated user's encrypted BYOK credential.
- Ollama runs locally through the Local Connector on `127.0.0.1:8765`.
- Memory retrieval is user-scoped, bounded, and controlled by persisted settings.
- PostgreSQL stores users, conversations, messages, memories, credentials, and model registry data.
- ChromaDB stores searchable memory vectors.

## Run

```powershell
pnpm install
pnpm dev
python -m uvicorn app.main:app --app-dir server --reload
python -m uvicorn local_connector.main:app --app-dir server --host 127.0.0.1 --port 8765
```

Run migrations with `alembic -c server/alembic.ini upgrade head` from the repository root.

See [BYOK_SETUP.md](BYOK_SETUP.md), [LOCAL_OLLAMA_SETUP.md](LOCAL_OLLAMA_SETUP.md), and [API_DOCUMENTATION.md](API_DOCUMENTATION.md).

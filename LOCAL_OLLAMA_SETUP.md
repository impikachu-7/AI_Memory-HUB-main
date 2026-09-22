# Local Ollama Setup

The local connector keeps Ollama on the user's computer. The deployed FastAPI service never connects to a user's `localhost`.

1. Install Ollama from https://ollama.com/download.
2. Pull a model, for example `ollama pull qwen2.5:7b`.
3. From the repository root, install server dependencies and run:

```powershell
python -m uvicorn local_connector.main:app --app-dir server --host 127.0.0.1 --port 8765
```

4. For the deployed app, allow its exact Vercel origin, for example `https://ai-memory-hub-phi.vercel.app`:

```powershell
$env:CONNECTOR_ALLOWED_ORIGINS="https://ai-memory-hub-phi.vercel.app"
```

5. Verify `http://127.0.0.1:8765/health` reports `connected` and `/models` lists the locally installed models.
6. In the deployed web app, choose the discovered Ollama model. Chat generation runs in the browser through the local connector; the deployed backend only prepares memory/context and persists the final response.

The connector defaults to `127.0.0.1:11434`, accepts browser origins from `CONNECTOR_ALLOWED_ORIGINS`, and does not receive cloud provider keys.

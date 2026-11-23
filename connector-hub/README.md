# connector-hub

Let any AI use your apps. An MCP server with OAuth for Google (Gmail, Calendar, Drive, YouTube), Instagram and TikTok. Tokens stay on your machine (`~/.connector-hub/tokens.json`, mode 600) and refresh automatically.

Works with any MCP client: Claude (Code / Desktop / claude.ai), ChatGPT, Cursor, Codex and others.

## Tools (13)

| App | Tools |
|---|---|
| Gmail | `gmail_search`, `gmail_send` |
| Calendar | `calendar_upcoming`, `calendar_create_event` |
| Drive | `drive_search` |
| YouTube | `youtube_search`, `youtube_my_videos` |
| Instagram | `instagram_profile`, `instagram_my_posts`, `instagram_post_photo` |
| TikTok | `tiktok_profile`, `tiktok_my_videos` |
| — | `list_connections` |

## Setup

```bash
pip install -r requirements.txt
```

### 1. Create a developer app per provider (one-time)

Each provider gives you a client ID and secret. Put them in env vars named `<APP>_CLIENT_ID` and `<APP>_CLIENT_SECRET`.

- **Google**: [console.cloud.google.com](https://console.cloud.google.com) → new project → enable Gmail, Calendar, Drive, YouTube Data APIs → OAuth consent screen (External, Testing, add yourself as test user) → Credentials → OAuth client ID, type **Web application**, redirect URI `http://localhost:8765/callback`.
- **Instagram**: [developers.facebook.com](https://developers.facebook.com) → create app → add **Instagram** product → "API setup with Instagram login". Needs an Instagram **Business or Creator** account. Redirect URI must be https (see below).
- **TikTok**: [developers.tiktok.com](https://developers.tiktok.com) → create app → add **Login Kit** → scopes `user.info.basic`, `video.list`. Redirect URI must be https (see below).

**https redirect for Instagram/TikTok:** run a tunnel to port 8765 (e.g. `cloudflared tunnel --url http://localhost:8765`), register `https://<tunnel-host>/callback` with the provider, and set `HUB_REDIRECT_URI` to the same URL before running `connect`.

### 2. Connect your accounts

```bash
python -m hub connect google
python -m hub connect instagram
python -m hub connect tiktok
```

### 3. Plug into your AI

**Claude Code**
```bash
claude mcp add connector-hub -e PYTHONPATH=/path/to/connector-hub -- python -m hub serve
```

**Claude Desktop / Cursor** (`claude_desktop_config.json` / `.cursor/mcp.json`):
```json
{ "mcpServers": { "connector-hub": { "command": "python", "args": ["-m", "hub", "serve"], "env": { "PYTHONPATH": "/path/to/connector-hub" } } } }
```
The env vars with your client IDs/secrets must be visible to the server too (add them to `env`) so tokens can refresh.

**ChatGPT, claude.ai and other remote clients** need an HTTPS URL:
```bash
python -m hub serve --http          # http://127.0.0.1:8000/mcp
```
then expose it with a tunnel. **Warning:** the HTTP endpoint has no login yet. Anyone with the URL can read your mail and post as you. Only expose it briefly, and never share the URL.

## Add an app

1. Add its OAuth settings to `hub/apps.py`.
2. Add `@mcp.tool()` functions in `hub/server.py` that call `api("<app>", ...)`.

## Test

```bash
python test_hub.py    # offline, no accounts needed
```

## Limits to know

- Google apps stay in **Testing** mode: max 100 test users, tokens expire every 7 days. Going public needs Google verification, and Gmail scopes need a paid security assessment (CASA).
- Instagram/TikTok apps work for your own accounts in dev mode. Other users need Meta App Review / TikTok audit. TikTok posting (`video.publish`) needs an audit before posts go public.

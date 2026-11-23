# connector-hub

Let any AI use your apps. An MCP server for Google (Gmail, Calendar, Drive, YouTube), Instagram, TikTok, Telegram, Discord, Slack, GitHub and Notion. Tokens stay on your machine (`~/.connector-hub/tokens.json`, mode 600) and refresh automatically.

Works with any MCP client: Claude (Code / Desktop / claude.ai), ChatGPT, Cursor, Codex and others.

## Tools (34)

| App | Tools |
|---|---|
| Gmail | `gmail_search`, `gmail_send` |
| Calendar | `calendar_upcoming`, `calendar_create_event` |
| Drive | `drive_search` |
| YouTube | `youtube_search`, `youtube_my_videos`, `youtube_trending`, `youtube_video_details` |
| Instagram | `instagram_profile`, `instagram_my_posts`, `instagram_post_photo`, `instagram_post_reel`, `instagram_post_carousel`, `instagram_post_story`, `instagram_comments`, `instagram_reply_comment`, `instagram_post_insights` |
| TikTok | `tiktok_profile`, `tiktok_my_videos`, `tiktok_post_video`, `tiktok_post_status` |
| Telegram | `telegram_recent_messages`, `telegram_send` |
| Discord | `discord_read`, `discord_send` |
| Slack | `slack_read`, `slack_send` |
| GitHub | `github_my_repos`, `github_issues`, `github_create_issue` |
| Notion | `notion_search`, `notion_create_page` |
| — | `list_connections` |

Instagram posting needs a **public URL** for each image/video (e.g. a public Drive/S3/Cloudinary link). TikTok uploads a **local file** (up to 64 MB).

The server tells the AI to show you every caption and reply and get a yes before posting.

## Setup

```bash
pip install -r requirements.txt
```

### 1. Create a developer app per provider (one-time)

Each provider gives you a client ID and secret. Put them in env vars named `<APP>_CLIENT_ID` and `<APP>_CLIENT_SECRET`.

- **Google**: [console.cloud.google.com](https://console.cloud.google.com) → new project → enable Gmail, Calendar, Drive, YouTube Data APIs → OAuth consent screen (External, Testing, add yourself as test user) → Credentials → OAuth client ID, type **Web application**, redirect URI `http://localhost:8765/callback`.
- **Instagram**: [developers.facebook.com](https://developers.facebook.com) → create app → add **Instagram** product → "API setup with Instagram login". Needs an Instagram **Business or Creator** account. Redirect URI must be https (see below).
- **TikTok**: [developers.tiktok.com](https://developers.tiktok.com) → create app → add **Login Kit** → scopes `user.info.basic`, `video.list`; add **Content Posting API** → `video.publish`. Redirect URI must be https (see below).

**https redirect for Instagram/TikTok:** run a tunnel to port 8765 (e.g. `cloudflared tunnel --url http://localhost:8765`), register `https://<tunnel-host>/callback` with the provider, and set `HUB_REDIRECT_URI` to the same URL before running `connect`.

### Token apps (no OAuth, just set an env var)

| App | Env var | Where to get it |
|---|---|---|
| Telegram | `TELEGRAM_TOKEN` | Message @BotFather → `/newbot`. Add the bot to your group/channel. |
| Discord | `DISCORD_TOKEN` | [discord.com/developers](https://discord.com/developers/applications) → New App → Bot → Reset Token; enable **Message Content Intent**; invite the bot to your server with Read/Send Messages. |
| Slack | `SLACK_TOKEN` | [api.slack.com/apps](https://api.slack.com/apps) → Create App → OAuth & Permissions → bot scopes `chat:write`, `channels:history` → Install → copy the `xoxb-` token. `/invite` the bot to channels. |
| GitHub | `GITHUB_TOKEN` | GitHub → Settings → Developer settings → Fine-grained token with Issues read/write on chosen repos. |
| Notion | `NOTION_TOKEN` | [notion.so/my-integrations](https://www.notion.so/my-integrations) → New integration → copy secret. On each page: ••• → Connections → add it. |

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
- Instagram/TikTok apps work for your own accounts in dev mode. Other users need Meta App Review / TikTok audit.
- TikTok posts are **private (SELF_ONLY)** until TikTok audits your app.
- TikTok has no public API for replying to comments, and Instagram Login has no hashtag search, so trend research uses YouTube.

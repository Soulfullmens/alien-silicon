"""MCP server exposing app tools. Any MCP client (Claude, ChatGPT, Cursor, Codex...) can use it."""

import base64
import io
import json
import mimetypes
import os
import secrets
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path

import httpx
from PIL import Image
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .auth import HOME, connected, get_token

mcp = MCPServer("connector-hub", instructions="""Tools for the user's Gmail, Calendar, Drive, YouTube, Instagram, TikTok,
Telegram, Discord, Slack, GitHub and Notion, plus a trend-watching and content studio.
Trend research: analyze_recording on the user's screen recordings of scrolling Reels/TikTok (iPhone recordings:
find them with drive_search), then trending for "what's trending today". Also youtube_trending, and the user's own
best posts (instagram_my_posts + instagram_post_insights, tiktok_my_videos).
Creating: use the trends to write an original idea, script and caption (hook in the first line, call to action,
3-8 relevant hashtags), then generate_image / generate_video. Never re-upload other creators' videos.
ALWAYS show the user the draft (caption + media path) and get a yes before posting or replying to anyone.
Post: Instagram needs public_url (instagram_post_story / _reel / _photo); TikTok takes file_path.""")


def token(app):
    # ToolError (not RuntimeError) so the message reaches the AI instead of a generic "Error executing tool"
    try:
        return get_token(app)
    except RuntimeError as e:
        raise ToolError(str(e)) from e


def api(app, method, url, headers=None, **kw):
    scheme = "Bot" if app == "discord" else "Bearer"
    headers = {"Authorization": f"{scheme} {token(app)}", **(headers or {})}
    r = httpx.request(method, url, headers=headers, timeout=60, **kw)
    if r.is_error:
        raise ToolError(f"{app} API {r.status_code}: {r.text[:500]}")
    return r.json()


@mcp.tool()
def list_connections() -> list[str]:
    """Apps the user has connected."""
    return connected()


# ---------- Google ----------
G = "https://www.googleapis.com"


@mcp.tool()
def gmail_search(query: str = "", max_results: int = 10) -> list[dict]:
    """Search Gmail (Gmail search syntax, e.g. 'from:bob is:unread'). Returns id, from, subject, snippet."""
    ids = api("google", "GET", f"{G}/gmail/v1/users/me/messages", params={"q": query, "maxResults": max_results})
    out = []
    for m in ids.get("messages", []):
        msg = api("google", "GET", f"{G}/gmail/v1/users/me/messages/{m['id']}",
                  params={"format": "metadata", "metadataHeaders": ["From", "Subject", "Date"]})
        h = {x["name"]: x["value"] for x in msg["payload"].get("headers", [])}
        out.append({"id": m["id"], "from": h.get("From"), "subject": h.get("Subject"),
                    "date": h.get("Date"), "snippet": msg.get("snippet")})
    return out


@mcp.tool()
def gmail_send(to: str, subject: str, body: str) -> dict:
    """Send a plain-text email from the user's Gmail."""
    m = EmailMessage()
    m["To"], m["Subject"] = to, subject
    m.set_content(body)
    raw = base64.urlsafe_b64encode(m.as_bytes()).decode()
    return api("google", "POST", f"{G}/gmail/v1/users/me/messages/send", json={"raw": raw})


@mcp.tool()
def calendar_upcoming(max_results: int = 10) -> list[dict]:
    """Upcoming events on the user's primary Google Calendar."""
    r = api("google", "GET", f"{G}/calendar/v3/calendars/primary/events", params={
        "timeMin": datetime.now(timezone.utc).isoformat(), "singleEvents": "true",
        "orderBy": "startTime", "maxResults": max_results})
    return [{"id": e["id"], "summary": e.get("summary"), "start": e["start"], "end": e["end"]}
            for e in r.get("items", [])]


@mcp.tool()
def calendar_create_event(summary: str, start: str, end: str, description: str = "") -> dict:
    """Create an event. start/end are ISO 8601 with timezone, e.g. 2026-11-25T10:00:00+05:30."""
    return api("google", "POST", f"{G}/calendar/v3/calendars/primary/events", json={
        "summary": summary, "description": description,
        "start": {"dateTime": start}, "end": {"dateTime": end}})


@mcp.tool()
def drive_search(name_contains: str, max_results: int = 10) -> list[dict]:
    """Find Google Drive files whose name contains the text."""
    q = name_contains.replace("\\", "\\\\").replace("'", "\\'")
    r = api("google", "GET", f"{G}/drive/v3/files", params={
        "q": f"name contains '{q}' and trashed = false", "pageSize": max_results,
        "fields": "files(id,name,mimeType,webViewLink,modifiedTime)"})
    return r.get("files", [])


@mcp.tool()
def youtube_search(query: str, max_results: int = 10) -> list[dict]:
    """Search YouTube videos."""
    r = api("google", "GET", f"{G}/youtube/v3/search", params={
        "part": "snippet", "q": query, "type": "video", "maxResults": max_results})
    return [{"video_id": i["id"]["videoId"], "title": i["snippet"]["title"],
             "channel": i["snippet"]["channelTitle"]} for i in r.get("items", [])]


@mcp.tool()
def youtube_my_videos(max_results: int = 10) -> list[dict]:
    """The user's own uploaded YouTube videos, newest first."""
    ch = api("google", "GET", f"{G}/youtube/v3/channels", params={"part": "contentDetails", "mine": "true"})
    if not ch.get("items"):
        return []
    uploads = ch["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    r = api("google", "GET", f"{G}/youtube/v3/playlistItems", params={
        "part": "snippet", "playlistId": uploads, "maxResults": max_results})
    return [{"video_id": i["snippet"]["resourceId"]["videoId"], "title": i["snippet"]["title"],
             "published": i["snippet"]["publishedAt"]} for i in r.get("items", [])]


@mcp.tool()
def youtube_trending(region_code: str = "IN", category_id: str = "", max_results: int = 20) -> list[dict]:
    """Most popular YouTube videos right now in a country (ISO code, e.g. IN, US). category_id e.g. 10=Music, 24=Entertainment."""
    params = {"part": "snippet,statistics", "chart": "mostPopular", "regionCode": region_code, "maxResults": max_results}
    if category_id:
        params["videoCategoryId"] = category_id
    r = api("google", "GET", f"{G}/youtube/v3/videos", params=params)
    return [_yt(v) for v in r.get("items", [])]


@mcp.tool()
def youtube_video_details(video_id: str) -> dict:
    """Title, description, tags and view/like/comment counts of any YouTube video."""
    r = api("google", "GET", f"{G}/youtube/v3/videos", params={"part": "snippet,statistics", "id": video_id})
    if not r.get("items"):
        raise ToolError(f"No video {video_id}")
    return _yt(r["items"][0], full=True)


def _yt(v, full=False):
    s, st = v["snippet"], v.get("statistics", {})
    d = {"video_id": v["id"], "title": s["title"], "channel": s["channelTitle"], "published": s["publishedAt"],
         "views": st.get("viewCount"), "likes": st.get("likeCount"), "comments": st.get("commentCount"),
         "tags": s.get("tags", [])}
    if full:
        d["description"] = s.get("description")
    return d


# ---------- Instagram ----------
IG = "https://graph.instagram.com"  # unversioned = app's default Graph version
POLL_SECONDS = 5


def _ig_publish(params):
    """Create a media container, wait until Instagram has processed it, then publish."""
    c = api("instagram", "POST", f"{IG}/me/media", params=params)["id"]
    for _ in range(60):  # videos can take a few minutes
        status = api("instagram", "GET", f"{IG}/{c}", params={"fields": "status_code"}).get("status_code")
        if status == "FINISHED":
            return api("instagram", "POST", f"{IG}/me/media_publish", params={"creation_id": c})
        if status in ("ERROR", "EXPIRED"):
            raise ToolError(f"Instagram could not process the media (status {status}).")
        time.sleep(POLL_SECONDS)
    raise ToolError("Instagram is still processing the media after 5 minutes; try again later.")


@mcp.tool()
def instagram_profile() -> dict:
    """The connected Instagram account's profile."""
    return api("instagram", "GET", f"{IG}/me",
               params={"fields": "user_id,username,account_type,media_count,followers_count"})


@mcp.tool()
def instagram_my_posts(limit: int = 10) -> list[dict]:
    """Recent posts on the connected Instagram account, with like and comment counts."""
    r = api("instagram", "GET", f"{IG}/me/media", params={
        "fields": "id,caption,media_type,permalink,timestamp,like_count,comments_count", "limit": limit})
    return r.get("data", [])


@mcp.tool()
def instagram_post_photo(image_url: str, caption: str = "") -> dict:
    """Publish a feed photo. image_url must be a public JPEG URL."""
    return _ig_publish({"image_url": image_url, "caption": caption})


@mcp.tool()
def instagram_post_reel(video_url: str, caption: str = "", share_to_feed: bool = True) -> dict:
    """Publish a Reel. video_url must be a public MP4 URL (9:16, 3s-15min)."""
    return _ig_publish({"media_type": "REELS", "video_url": video_url, "caption": caption,
                        "share_to_feed": str(share_to_feed).lower()})


@mcp.tool()
def instagram_post_carousel(image_urls: list[str], caption: str = "") -> dict:
    """Publish a carousel of 2-10 public JPEG URLs."""
    if not 2 <= len(image_urls) <= 10:
        raise ToolError("A carousel needs 2 to 10 images.")
    kids = [api("instagram", "POST", f"{IG}/me/media", params={"image_url": u, "is_carousel_item": "true"})["id"]
            for u in image_urls]
    return _ig_publish({"media_type": "CAROUSEL", "children": ",".join(kids), "caption": caption})


@mcp.tool()
def instagram_post_story(image_url: str = "", video_url: str = "") -> dict:
    """Publish a Story from one public image (JPEG) or video (MP4) URL."""
    if bool(image_url) == bool(video_url):
        raise ToolError("Give exactly one of image_url or video_url.")
    media = {"image_url": image_url} if image_url else {"video_url": video_url}
    return _ig_publish({"media_type": "STORIES", **media})


@mcp.tool()
def instagram_comments(media_id: str, limit: int = 25) -> list[dict]:
    """Comments on one of the user's posts (with replies)."""
    r = api("instagram", "GET", f"{IG}/{media_id}/comments", params={
        "fields": "id,text,username,timestamp,like_count,replies{id,text,username}", "limit": limit})
    return r.get("data", [])


@mcp.tool()
def instagram_reply_comment(comment_id: str, message: str) -> dict:
    """Reply to a comment on the user's post."""
    return api("instagram", "POST", f"{IG}/{comment_id}/replies", params={"message": message})


@mcp.tool()
def instagram_post_insights(media_id: str) -> list[dict]:
    """How a post performed: views, reach, likes, comments, shares, saves."""
    r = api("instagram", "GET", f"{IG}/{media_id}/insights",
            params={"metric": "views,reach,likes,comments,shares,saved"})
    return [{"metric": m["name"], "value": m["values"][0]["value"]} for m in r.get("data", [])]


# ---------- TikTok ----------
TT = "https://open.tiktokapis.com/v2"


@mcp.tool()
def tiktok_profile() -> dict:
    """The connected TikTok account's profile."""
    return api("tiktok", "GET", f"{TT}/user/info/", params={"fields": "open_id,display_name,avatar_url"})["data"]


@mcp.tool()
def tiktok_my_videos(max_count: int = 10) -> list[dict]:
    """Recent videos on the connected TikTok account (max 20), with view/like/comment/share counts."""
    r = api("tiktok", "POST", f"{TT}/video/list/",
            params={"fields": "id,title,create_time,share_url,view_count,like_count,comment_count,share_count"},
            json={"max_count": min(max_count, 20)})
    return r["data"].get("videos", [])


@mcp.tool()
def tiktok_post_video(file_path: str, title: str, privacy_level: str = "SELF_ONLY") -> dict:
    """Post a local MP4 to TikTok. title = caption with #hashtags.
    Until the TikTok app passes audit, only privacy_level SELF_ONLY works."""
    info = api("tiktok", "POST", f"{TT}/post/publish/creator_info/query/", json={})["data"]
    if privacy_level not in info["privacy_level_options"]:
        raise ToolError(f"privacy_level must be one of {info['privacy_level_options']}")
    size = os.path.getsize(file_path)
    # shortcut: single-chunk upload caps files at 64 MB; add chunking if longer videos are needed
    if size > 64 * 1024 * 1024:
        raise ToolError("Video is over 64 MB; compress it first.")
    init = api("tiktok", "POST", f"{TT}/post/publish/video/init/", json={
        "post_info": {"title": title, "privacy_level": privacy_level},
        "source_info": {"source": "FILE_UPLOAD", "video_size": size, "chunk_size": size, "total_chunk_count": 1},
    })["data"]
    with open(file_path, "rb") as f:
        r = httpx.put(init["upload_url"], content=f.read(), timeout=300, headers={
            "Content-Type": "video/mp4", "Content-Range": f"bytes 0-{size - 1}/{size}"})
    if r.is_error:
        raise ToolError(f"TikTok upload {r.status_code}: {r.text[:300]}")
    return {"publish_id": init["publish_id"], "next": "call tiktok_post_status with this publish_id"}


@mcp.tool()
def tiktok_post_status(publish_id: str) -> dict:
    """Check whether a TikTok post finished processing."""
    return api("tiktok", "POST", f"{TT}/post/publish/status/fetch/", json={"publish_id": publish_id})["data"]


# ---------- Telegram (bot) ----------
def _tg(method, **params):
    r = httpx.post(f"https://api.telegram.org/bot{token('telegram')}/{method}", json=params, timeout=30).json()
    if not r.get("ok"):
        raise ToolError(f"telegram: {r.get('description')}")
    return r["result"]


@mcp.tool()
def telegram_recent_messages(limit: int = 20) -> list[dict]:
    """Recent messages sent to the bot (shows chat_id to reply to)."""
    return [{"chat_id": u["message"]["chat"]["id"], "from": u["message"].get("from", {}).get("username"),
             "text": u["message"].get("text"), "date": u["message"]["date"]}
            for u in _tg("getUpdates", limit=limit) if "message" in u]


@mcp.tool()
def telegram_send(chat_id: str, text: str) -> dict:
    """Send a message from the bot to a chat, group or channel (@channelname works for channels)."""
    m = _tg("sendMessage", chat_id=chat_id, text=text)
    return {"message_id": m["message_id"]}


# ---------- Discord (bot) ----------
DC = "https://discord.com/api/v10"


@mcp.tool()
def discord_read(channel_id: str, limit: int = 20) -> list[dict]:
    """Recent messages in a Discord channel the bot can see."""
    return [{"id": m["id"], "author": m["author"]["username"], "content": m["content"], "time": m["timestamp"]}
            for m in api("discord", "GET", f"{DC}/channels/{channel_id}/messages", params={"limit": limit})]


@mcp.tool()
def discord_send(channel_id: str, content: str) -> dict:
    """Post a message to a Discord channel as the bot."""
    return {"id": api("discord", "POST", f"{DC}/channels/{channel_id}/messages", json={"content": content})["id"]}


# ---------- Slack (bot) ----------
def _slack(method, **kw):
    r = api("slack", "POST", f"https://slack.com/api/{method}", **kw)
    if not r.get("ok"):  # Slack returns HTTP 200 with ok:false on errors
        raise ToolError(f"slack: {r.get('error')}")
    return r


@mcp.tool()
def slack_read(channel_id: str, limit: int = 20) -> list[dict]:
    """Recent messages in a Slack channel the bot is in."""
    r = _slack("conversations.history", data={"channel": channel_id, "limit": limit})
    return [{"user": m.get("user"), "text": m.get("text"), "ts": m["ts"]} for m in r["messages"]]


@mcp.tool()
def slack_send(channel_id: str, text: str) -> dict:
    """Post a message to a Slack channel as the bot."""
    return {"ts": _slack("chat.postMessage", json={"channel": channel_id, "text": text})["ts"]}


# ---------- GitHub ----------
GH = "https://api.github.com"


@mcp.tool()
def github_my_repos(limit: int = 20) -> list[dict]:
    """The user's GitHub repos, most recently updated first."""
    return [{"name": r["full_name"], "private": r["private"], "url": r["html_url"]}
            for r in api("github", "GET", f"{GH}/user/repos", params={"sort": "updated", "per_page": limit})]


@mcp.tool()
def github_issues(repo: str, state: str = "open", limit: int = 20) -> list[dict]:
    """Issues in a repo ('owner/name')."""
    return [{"number": i["number"], "title": i["title"], "state": i["state"], "url": i["html_url"]}
            for i in api("github", "GET", f"{GH}/repos/{repo}/issues", params={"state": state, "per_page": limit})
            if "pull_request" not in i]


@mcp.tool()
def github_create_issue(repo: str, title: str, body: str = "") -> dict:
    """Open an issue in a repo ('owner/name')."""
    i = api("github", "POST", f"{GH}/repos/{repo}/issues", json={"title": title, "body": body})
    return {"number": i["number"], "url": i["html_url"]}


# ---------- Notion ----------
NO = "https://api.notion.com/v1"
NOTION_HEADERS = {"Notion-Version": "2022-06-28"}


@mcp.tool()
def notion_search(query: str, limit: int = 10) -> list[dict]:
    """Search Notion pages shared with the integration."""
    r = api("notion", "POST", f"{NO}/search", headers=NOTION_HEADERS, json={"query": query, "page_size": limit})
    out = []
    for p in r.get("results", []):
        title = next((v["title"] for v in p.get("properties", {}).values() if v.get("type") == "title"), [])
        out.append({"id": p["id"], "type": p["object"], "url": p.get("url"),
                    "title": "".join(t["plain_text"] for t in title)})
    return out


@mcp.tool()
def notion_create_page(parent_page_id: str, title: str, text: str = "") -> dict:
    """Create a page under a parent page. text becomes paragraphs (one per line)."""
    blocks = [{"type": "paragraph", "paragraph": {"rich_text": [{"type": "text", "text": {"content": line[:2000]}}]}}
              for line in text.splitlines() if line.strip()]
    p = api("notion", "POST", f"{NO}/pages", headers=NOTION_HEADERS, json={
        "parent": {"page_id": parent_page_id},
        "properties": {"title": {"title": [{"text": {"content": title}}]}},
        "children": blocks[:100]})
    return {"id": p["id"], "url": p["url"]}


# ---------- Gemini: watch screen recordings, track trends, generate media ----------
GM = "https://generativelanguage.googleapis.com"
# Model IDs change often. "-latest" alias tracks the newest Flash; generation models can be overridden per call
# (use gemini_models to see what's live). shortcut: generation request shapes not tested against the live API yet.
WATCH_MODEL = "gemini-flash-latest"
IMAGE_MODEL = "gemini-2.5-flash-image"
VIDEO_MODEL = "veo-3.1-generate-preview"
MEDIA = HOME / "media"
REELS = HOME / "reels.jsonl"

REEL_FIELDS = ["platform", "creator", "caption", "audio", "likes", "comments", "shares", "on_screen_text",
               "spoken_words", "hook", "format", "topic", "niche", "why_it_works"]
REEL_SCHEMA = {"type": "ARRAY", "items": {"type": "OBJECT", "required": ["platform", "creator", "hook", "format", "topic"],
               "properties": {**{f: {"type": "STRING"} for f in REEL_FIELDS},
                              "hashtags": {"type": "ARRAY", "items": {"type": "STRING"}},
                              "is_ad": {"type": "BOOLEAN"}, "start_seconds": {"type": "NUMBER"}}}}
WATCH_PROMPT = """This is a screen recording of someone scrolling Instagram Reels or TikTok.
For EVERY distinct reel/video shown, return one object:
platform (Instagram or TikTok), creator (@username as shown), caption, hashtags, audio (song/sound name shown),
likes, comments, shares (exactly as displayed, e.g. "12.3K"), on_screen_text, spoken_words (short summary of
speech/lyrics you hear), hook (what happens in the first 3 seconds that makes people stay), format (e.g. talking head,
POV skit, tutorial, before/after, slideshow, meme, dance), topic, niche, why_it_works (one sentence),
is_ad, start_seconds (when it appears in the recording).
Only report what is visible or audible; use "" when something is not shown."""


def _gk():
    if not os.environ.get("GEMINI_API_KEY"):
        raise ToolError("Set GEMINI_API_KEY (free key at aistudio.google.com).")
    return {"x-goog-api-key": os.environ["GEMINI_API_KEY"]}


def _gm(method, path, **kw):
    r = httpx.request(method, GM + path, headers={**_gk(), **kw.pop("headers", {})}, timeout=kw.pop("timeout", 120), **kw)
    if r.is_error:
        raise ToolError(f"Gemini {r.status_code}: {r.text[:500]}")
    return r


def _gm_upload(path, mime):
    size = os.path.getsize(path)
    start = _gm("POST", "/upload/v1beta/files", json={"file": {"display_name": os.path.basename(path)}}, headers={
        "X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(size), "X-Goog-Upload-Header-Content-Type": mime})
    with open(path, "rb") as f:  # streamed: recordings can be hundreds of MB
        r = httpx.request("POST", start.headers["x-goog-upload-url"], timeout=None,
                          content=iter(lambda: f.read(1 << 20), b""), headers={
                              "Content-Length": str(size), "X-Goog-Upload-Offset": "0",
                              "X-Goog-Upload-Command": "upload, finalize"})
    if r.is_error:
        raise ToolError(f"Gemini upload {r.status_code}: {r.text[:300]}")
    f = r.json()["file"]
    for _ in range(120):
        if f["state"] != "PROCESSING":
            break
        time.sleep(POLL_SECONDS)
        f = _gm("GET", f"/v1beta/{f['name']}").json()
    if f["state"] != "ACTIVE":
        raise ToolError(f"Gemini could not process the video (state {f['state']}).")
    return f


def _drive_download(file_id):
    meta = api("google", "GET", f"{G}/drive/v3/files/{file_id}", params={"fields": "name,mimeType"})
    rec_dir = HOME / "recordings"  # private: never inside the publicly served MEDIA folder
    rec_dir.mkdir(parents=True, exist_ok=True)
    dest = rec_dir / f"{secrets.token_hex(8)}{Path(meta['name']).suffix or '.mp4'}"
    with httpx.stream("GET", f"{G}/drive/v3/files/{file_id}", params={"alt": "media"}, timeout=None,
                      headers={"Authorization": f"Bearer {token('google')}"}, follow_redirects=True) as r:
        if r.is_error:
            raise ToolError(f"Drive download {r.status_code}")
        with open(dest, "wb") as out:
            for chunk in r.iter_bytes():
                out.write(chunk)
    return str(dest), meta["mimeType"]


@mcp.tool()
def analyze_recording(file_path: str = "", drive_file_id: str = "", delete_after: bool = True) -> dict:
    """Watch + listen to a screen recording of scrolling Reels/TikTok and save details of every video in it.
    Give a local file_path (Android/scrcpy) or a drive_file_id (iPhone recording uploaded to Google Drive).
    delete_after removes the local recording once the details are saved, to free space."""
    if bool(file_path) == bool(drive_file_id):
        raise ToolError("Give exactly one of file_path or drive_file_id.")
    if drive_file_id:
        file_path, mime = _drive_download(drive_file_id)
        delete_after = True  # our own temp copy; the original stays in Drive
    else:
        mime = mimetypes.guess_type(file_path)[0] or "video/mp4"
    f = _gm_upload(file_path, mime)
    try:
        r = _gm("POST", f"/v1beta/models/{WATCH_MODEL}:generateContent", timeout=600, json={
            "contents": [{"parts": [{"file_data": {"mime_type": mime, "file_uri": f["uri"]}}, {"text": WATCH_PROMPT}]}],
            "generationConfig": {"responseMimeType": "application/json", "responseSchema": REEL_SCHEMA}}).json()
    finally:
        _gm("DELETE", f"/v1beta/{f['name']}")  # don't leave the recording on Google's side
    reels = json.loads(r["candidates"][0]["content"]["parts"][0]["text"])
    seen = datetime.now(timezone.utc).isoformat()
    HOME.mkdir(parents=True, exist_ok=True)
    with open(REELS, "a") as out:
        for reel in reels:
            out.write(json.dumps({**reel, "seen_at": seen}) + "\n")
    if delete_after:
        os.remove(file_path)  # only after the details are safely written
    return {"videos_found": len(reels), "videos": reels}


def _num(s):
    """'12.3K' -> 12300, '1.2M' -> 1200000, '1,234' -> 1234, '' -> 0."""
    s = str(s or "").strip().upper().replace(",", "")
    mult = {"K": 1e3, "M": 1e6, "B": 1e9}.get(s[-1:], 1)
    try:
        return int(float(s.rstrip("KMB")) * mult)
    except ValueError:
        return 0


@mcp.tool()
def trending(days: int = 1, platform: str = "") -> dict:
    """What's trending in the videos watched over the last N days: top hashtags, sounds, formats, topics, hooks
    and the most-liked videos. platform: Instagram, TikTok or "" for both."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    rows = [json.loads(line) for line in REELS.read_text().splitlines()] if REELS.exists() else []
    rows = [r for r in rows if r["seen_at"] >= cutoff and (not platform or r.get("platform", "").lower() == platform.lower())]

    def top(key, n=10):
        return Counter(v for r in rows for v in (r.get(key) if isinstance(r.get(key), list) else [r.get(key)])
                       if v).most_common(n)
    return {"videos_seen": len(rows), "top_hashtags": top("hashtags", 20), "top_audio": top("audio"),
            "top_formats": top("format"), "top_topics": top("topic"), "top_niches": top("niche"),
            "most_liked": sorted(rows, key=lambda r: _num(r.get("likes")), reverse=True)[:10]}


def _save_media(data, ext):
    MEDIA.mkdir(parents=True, exist_ok=True)
    name = f"{secrets.token_urlsafe(12)}.{ext}"  # unguessable: the media folder may be served publicly
    (MEDIA / name).write_bytes(data)
    base = os.environ.get("HUB_PUBLIC_URL", "").rstrip("/")
    return {"file_path": str(MEDIA / name), "public_url": f"{base}/{name}" if base else ""}


@mcp.tool()
def gemini_models() -> list[dict]:
    """Gemini models live on the user's key (use to pick a model for generate_image / generate_video)."""
    r = _gm("GET", "/v1beta/models", params={"pageSize": 1000}).json()
    return [{"id": m["name"].removeprefix("models/"), "methods": m.get("supportedGenerationMethods", [])}
            for m in r.get("models", [])]


@mcp.tool()
def generate_image(prompt: str, aspect_ratio: str = "9:16", model: str = IMAGE_MODEL) -> dict:
    """Generate an image (9:16 for Stories/Reels covers, 1:1 or 4:5 for feed). Returns file_path and public_url
    (public_url is set when the media server + tunnel are running; Instagram needs it)."""
    r = _gm("POST", f"/v1beta/models/{model}:generateContent", json={
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["IMAGE"], "imageConfig": {"aspectRatio": aspect_ratio}}}).json()
    parts = r.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    img = next((p.get("inlineData") or p.get("inline_data") for p in parts if p.get("inlineData") or p.get("inline_data")), None)
    if not img:
        raise ToolError(f"No image returned: {str(r)[:300]}")
    with Image.open(io.BytesIO(base64.b64decode(img["data"]))) as im:  # Instagram only accepts JPEG
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=92)
    return _save_media(buf.getvalue(), "jpg")


@mcp.tool()
def generate_video(prompt: str, aspect_ratio: str = "9:16", model: str = VIDEO_MODEL) -> dict:
    """Generate a short video (with sound) for Reels/TikTok/Stories. Takes 1-6 minutes.
    Returns file_path (for tiktok_post_video) and public_url (for instagram_post_reel / instagram_post_story)."""
    op = _gm("POST", f"/v1beta/models/{model}:predictLongRunning", json={
        "instances": [{"prompt": prompt}], "parameters": {"aspectRatio": aspect_ratio}}).json()
    for _ in range(72):
        if op.get("done"):
            break
        time.sleep(POLL_SECONDS * 2)
        op = _gm("GET", f"/v1beta/{op['name']}").json()
    if not op.get("done"):
        raise ToolError(f"Video still generating; operation {op['name']}")
    if "error" in op:
        raise ToolError(f"Video generation failed: {op['error']}")
    uri = op["response"]["generateVideoResponse"]["generatedSamples"][0]["video"]["uri"]
    data = httpx.request("GET", uri, headers=_gk(), follow_redirects=True, timeout=300)
    if data.is_error:
        raise ToolError(f"Video download {data.status_code}")
    return _save_media(data.content, "mp4")

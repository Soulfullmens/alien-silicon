"""MCP server exposing app tools. Any MCP client (Claude, ChatGPT, Cursor, Codex...) can use it."""

import base64
from datetime import datetime, timezone
from email.message import EmailMessage

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .auth import connected, get_token

mcp = MCPServer("connector-hub", instructions="Tools for the user's Gmail, Calendar, Drive, YouTube, Instagram and TikTok.")


def api(app, method, url, **kw):
    # ToolError (not RuntimeError) so the message reaches the AI instead of a generic "Error executing tool"
    try:
        token = get_token(app)
    except RuntimeError as e:
        raise ToolError(str(e)) from e
    r = httpx.request(method, url, headers={"Authorization": f"Bearer {token}"}, timeout=30, **kw)
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


# ---------- Instagram ----------
IG = "https://graph.instagram.com"  # unversioned = app's default Graph version


@mcp.tool()
def instagram_profile() -> dict:
    """The connected Instagram account's profile."""
    return api("instagram", "GET", f"{IG}/me", params={"fields": "user_id,username,account_type,media_count"})


@mcp.tool()
def instagram_my_posts(limit: int = 10) -> list[dict]:
    """Recent posts on the connected Instagram account."""
    r = api("instagram", "GET", f"{IG}/me/media", params={
        "fields": "id,caption,media_type,permalink,timestamp,like_count,comments_count", "limit": limit})
    return r.get("data", [])


@mcp.tool()
def instagram_post_photo(image_url: str, caption: str = "") -> dict:
    """Publish a photo. image_url must be a public JPEG URL."""
    c = api("instagram", "POST", f"{IG}/me/media", params={"image_url": image_url, "caption": caption})
    return api("instagram", "POST", f"{IG}/me/media_publish", params={"creation_id": c["id"]})


# ---------- TikTok ----------
TT = "https://open.tiktokapis.com/v2"


@mcp.tool()
def tiktok_profile() -> dict:
    """The connected TikTok account's profile."""
    return api("tiktok", "GET", f"{TT}/user/info/", params={"fields": "open_id,display_name,avatar_url"})["data"]


@mcp.tool()
def tiktok_my_videos(max_count: int = 10) -> list[dict]:
    """Recent videos on the connected TikTok account (max 20)."""
    r = api("tiktok", "POST", f"{TT}/video/list/", params={"fields": "id,title,create_time,share_url,view_count"},
            json={"max_count": min(max_count, 20)})
    return r["data"].get("videos", [])

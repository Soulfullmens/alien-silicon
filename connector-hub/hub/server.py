"""MCP server exposing app tools. Any MCP client (Claude, ChatGPT, Cursor, Codex...) can use it."""

import base64
import os
import time
from datetime import datetime, timezone
from email.message import EmailMessage

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .auth import connected, get_token

mcp = MCPServer("connector-hub", instructions="""Tools for the user's Gmail, Calendar, Drive, YouTube, Instagram, TikTok,
Telegram, Discord, Slack, GitHub and Notion.
Content workflow: 1) research: youtube_trending / youtube_video_details, plus the user's own best posts
(instagram_my_posts + instagram_post_insights, tiktok_my_videos); 2) write the caption yourself: hook in the
first line, short, a call to action, 3-8 relevant hashtags; 3) show the user the draft and get a yes
before posting anything; 4) post; 5) later, read comments and draft replies for the user to approve.""")


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

"""Offline self-check: python test_hub.py (no real accounts needed)."""

import asyncio
import os
import stat
import tempfile
import time
import urllib.parse

os.environ["HUB_HOME"] = tempfile.mkdtemp()
os.environ.update(GOOGLE_CLIENT_ID="cid", GOOGLE_CLIENT_SECRET="sec", TIKTOK_CLIENT_ID="tk", TIKTOK_CLIENT_SECRET="ts")

from hub import auth, server  # noqa: E402


class R:
    def __init__(self, data):
        self.data, self.is_error = data, False

    def json(self):
        return self.data


calls = []


def fake_post(url, data=None, **kw):
    calls.append((url, data))
    return R({"access_token": "new", "expires_in": 3600})


auth.httpx.post = fake_post

# auth URL carries the right client id param and scopes
q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(auth.auth_url("tiktok", "s1")).query))
assert q["client_key"] == "tk" and q["scope"] == "user.info.basic,video.list,video.publish" and q["state"] == "s1"
q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(auth.auth_url("google", "s2")).query))
assert q["access_type"] == "offline" and "gmail.send" in q["scope"]

# expired token is refreshed, refresh_token kept, file is private
auth._save({"google": {"access_token": "old", "refresh_token": "rt", "expires_at": time.time() - 1}})
assert auth.get_token("google") == "new"
assert calls[-1][1]["grant_type"] == "refresh_token" and calls[-1][1]["refresh_token"] == "rt"
assert auth._load()["google"]["refresh_token"] == "rt"
assert stat.S_IMODE(os.stat(auth.TOKENS).st_mode) == 0o600

# fresh token is not refreshed
n = len(calls)
assert auth.get_token("google") == "new" and len(calls) == n

# unconnected app gives a clear error
try:
    auth.get_token("instagram")
    raise AssertionError("expected error")
except RuntimeError as e:
    assert "connect instagram" in str(e)

# MCP server exposes the tools, and a tool call goes through with the stored token
names = {t.name for t in asyncio.run(server.mcp.list_tools())}
assert {"gmail_search", "gmail_send", "youtube_search", "instagram_post_photo", "tiktok_my_videos"} <= names, names

seen = {}


def fake_request(method, url, headers=None, **kw):
    seen.update(url=url, auth=headers["Authorization"], params=kw.get("params"))
    return R({"files": [{"id": "1", "name": "a'b"}]})


server.httpx.request = fake_request
assert server.drive_search("a'b") == [{"id": "1", "name": "a'b"}]
assert seen["auth"] == "Bearer new" and seen["params"]["q"] == "name contains 'a\\'b' and trashed = false"

# Instagram: waits for container processing, then publishes; ERROR status surfaces clearly
auth._save({**auth._load(), "instagram": {"access_token": "ig"}})
server.POLL_SECONDS = 0
script = []


def ig_request(method, url, headers=None, **kw):
    script.append((method, url.rsplit("/", 1)[-1], kw.get("params")))
    if url.endswith("/me/media"):
        return R({"id": "c1"})
    if url.endswith("/c1"):
        return R({"status_code": statuses.pop(0)})
    return R({"id": "post1"})


server.httpx.request = ig_request
statuses = ["IN_PROGRESS", "FINISHED"]
assert server.instagram_post_reel("https://x/v.mp4", "hi") == {"id": "post1"}
assert script[0][2]["media_type"] == "REELS" and script[-1][1] == "media_publish"
statuses = ["ERROR"]
try:
    server.instagram_post_story(image_url="https://x/a.jpg")
    raise AssertionError("expected error")
except server.ToolError as e:
    assert "ERROR" in str(e)
for bad in (lambda: server.instagram_post_story(), lambda: server.instagram_post_carousel(["one"])):
    try:
        bad()
        raise AssertionError("expected validation error")
    except server.ToolError:
        pass

# token apps: env var = connected; Discord uses "Bot" auth; Slack ok:false becomes an error
os.environ.update(DISCORD_TOKEN="dt", SLACK_TOKEN="st")
assert {"discord", "slack"} <= set(server.list_connections())
server.httpx.request = lambda m, u, headers=None, **kw: (seen.update(auth=headers["Authorization"]), R({"id": "9"}))[1]
assert server.discord_send("123", "hi") == {"id": "9"} and seen["auth"] == "Bot dt"
server.httpx.request = lambda m, u, headers=None, **kw: R({"ok": False, "error": "not_in_channel"})
try:
    server.slack_send("C1", "hi")
    raise AssertionError("expected error")
except server.ToolError as e:
    assert "not_in_channel" in str(e)

print(f"ok: {len(names)} tools")

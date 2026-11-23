"""OAuth connect flow + local token store with auto-refresh."""

import http.server
import json
import os
import secrets
import time
import urllib.parse
import webbrowser
from pathlib import Path

import httpx

from .apps import APPS

HOME = Path(os.environ.get("HUB_HOME", Path.home() / ".connector-hub"))
TOKENS = HOME / "tokens.json"
PORT = int(os.environ.get("HUB_PORT", "8765"))
# Google accepts http://localhost. Instagram/TikTok need an https URL: point a tunnel at PORT and set this.
REDIRECT_URI = os.environ.get("HUB_REDIRECT_URI", f"http://localhost:{PORT}/callback")


def _creds(app):
    p = app.upper()
    cid, secret = os.environ.get(f"{p}_CLIENT_ID"), os.environ.get(f"{p}_CLIENT_SECRET")
    if not cid or not secret:
        raise RuntimeError(f"Set {p}_CLIENT_ID and {p}_CLIENT_SECRET (from the {app} developer console).")
    return cid, secret


def _load():
    return json.loads(TOKENS.read_text()) if TOKENS.exists() else {}


def _save(all_tokens):
    HOME.mkdir(parents=True, exist_ok=True)
    # 0600: these tokens give full access to the user's accounts
    fd = os.open(TOKENS, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(all_tokens, f, indent=2)


def _store(app, tok, old=None):
    if "access_token" not in tok:
        raise RuntimeError(f"{app} token error: {tok}")
    entry = {**(old or {}), **tok}
    if "expires_in" in tok:
        entry["expires_at"] = time.time() + int(tok["expires_in"])
    all_tokens = _load()
    all_tokens[app] = entry
    _save(all_tokens)
    return entry


def auth_url(app, state):
    cfg = APPS[app]
    cid, _ = _creds(app)
    q = {
        cfg["client_id_param"]: cid,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": cfg["scope_sep"].join(cfg["scopes"]),
        "state": state,
        **cfg["extra_auth"],
    }
    return cfg["auth_url"] + "?" + urllib.parse.urlencode(q)


def _wait_for_code(state):
    result = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(self.path).query))
            if "code" not in q and "error" not in q:
                self.send_response(404)
                self.end_headers()
                return
            result.update(q)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Connected. You can close this tab.")

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", PORT), H)
    while not result:
        srv.handle_request()
    srv.server_close()
    if result.get("state") != state:  # CSRF check
        raise RuntimeError("OAuth state mismatch; aborting.")
    if "error" in result:
        raise RuntimeError(f"Provider refused: {result}")
    return result["code"]


def connect(app):
    if app not in APPS:
        raise SystemExit(f"Unknown app {app!r}. Known: {', '.join(APPS)}")
    cfg = APPS[app]
    cid, secret = _creds(app)
    state = secrets.token_urlsafe(16)
    url = auth_url(app, state)
    print(f"Open this URL to connect {app}:\n{url}")
    webbrowser.open(url)
    code = _wait_for_code(state)
    tok = httpx.post(cfg["token_url"], data={
        cfg["client_id_param"]: cid,
        "client_secret": secret,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": REDIRECT_URI,
    }).json()
    if app == "instagram":
        tok = tok["data"][0] if "data" in tok else tok
        # swap the 1-hour token for a 60-day one
        tok = httpx.get("https://graph.instagram.com/access_token", params={
            "grant_type": "ig_exchange_token", "client_secret": secret, "access_token": tok["access_token"],
        }).json()
    _store(app, tok)
    print(f"{app} connected. Token saved to {TOKENS}")


def _refresh(app, entry):
    if app == "instagram":
        tok = httpx.get("https://graph.instagram.com/refresh_access_token", params={
            "grant_type": "ig_refresh_token", "access_token": entry["access_token"],
        }).json()
    else:
        cfg = APPS[app]
        cid, secret = _creds(app)
        tok = httpx.post(cfg["token_url"], data={
            cfg["client_id_param"]: cid,
            "client_secret": secret,
            "grant_type": "refresh_token",
            "refresh_token": entry["refresh_token"],
        }).json()
    return _store(app, tok, old=entry)


def get_token(app):
    entry = _load().get(app)
    if not entry:
        raise RuntimeError(f"{app} is not connected. Run: python -m hub connect {app}")
    if entry.get("expires_at", float("inf")) - time.time() < 60:
        entry = _refresh(app, entry)
    return entry["access_token"]


def connected():
    return sorted(_load())

"""OAuth settings per provider. Add an app = add one entry here + its tools in server.py.

Token-only apps (telegram, discord, slack, github, notion) need no entry: set <APP>_TOKEN instead."""

APPS = {
    # One Google OAuth client covers Gmail, YouTube, Calendar and Drive.
    "google": {
        "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "scopes": [
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.send",
            "https://www.googleapis.com/auth/calendar.events",
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/youtube.readonly",
        ],
        "scope_sep": " ",
        "client_id_param": "client_id",
        # offline + consent = Google returns a refresh_token
        "extra_auth": {"access_type": "offline", "prompt": "consent"},
    },
    # Instagram API with Instagram Login (needs a Business or Creator account).
    "instagram": {
        "auth_url": "https://www.instagram.com/oauth/authorize",
        "token_url": "https://api.instagram.com/oauth/access_token",
        "scopes": [
            "instagram_business_basic",
            "instagram_business_content_publish",
            "instagram_business_manage_comments",
            "instagram_business_manage_insights",
        ],
        "scope_sep": ",",
        "client_id_param": "client_id",
        "extra_auth": {},
    },
    "tiktok": {
        "auth_url": "https://www.tiktok.com/v2/auth/authorize/",
        "token_url": "https://open.tiktokapis.com/v2/oauth/token/",
        "scopes": ["user.info.basic", "video.list", "video.publish"],
        "scope_sep": ",",
        "client_id_param": "client_key",  # TikTok calls the client id "client_key"
        "extra_auth": {},
    },
}

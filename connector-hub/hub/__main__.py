import sys

USAGE = "usage: python -m hub connect <google|instagram|tiktok> | serve [--http] | media"

args = sys.argv[1:]
if args[:1] == ["connect"] and len(args) == 2:
    from .auth import connect
    connect(args[1])
elif args[:1] == ["serve"]:
    from .server import mcp
    # --http: streamable HTTP on 127.0.0.1:8000/mcp (for ChatGPT, remote clients); default: stdio
    mcp.run("streamable-http") if "--http" in args else mcp.run()
elif args[:1] == ["media"]:
    # Serves generated media on 127.0.0.1:8766 so Instagram can fetch it through a tunnel (set HUB_PUBLIC_URL)
    import functools
    import http.server

    from .server import MEDIA

    class NoListing(http.server.SimpleHTTPRequestHandler):
        def list_directory(self, path):  # never reveal file names; only exact random names are fetchable
            self.send_error(404)

    MEDIA.mkdir(parents=True, exist_ok=True)
    print(f"Serving {MEDIA} on http://127.0.0.1:8766")
    http.server.ThreadingHTTPServer(("127.0.0.1", 8766), functools.partial(NoListing, directory=str(MEDIA))).serve_forever()
else:
    sys.exit(USAGE)

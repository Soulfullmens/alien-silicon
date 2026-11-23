import sys

USAGE = "usage: python -m hub connect <google|instagram|tiktok> | serve [--http]"

args = sys.argv[1:]
if args[:1] == ["connect"] and len(args) == 2:
    from .auth import connect
    connect(args[1])
elif args[:1] == ["serve"]:
    from .server import mcp
    # --http: streamable HTTP on 127.0.0.1:8000/mcp (for ChatGPT, remote clients); default: stdio
    mcp.run("streamable-http") if "--http" in args else mcp.run()
else:
    sys.exit(USAGE)

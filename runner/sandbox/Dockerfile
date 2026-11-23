# Judge sandbox for real agent runs (§22). The in-process fallback is judge/sandbox.py.
#   docker build -t aliensilicon-judge -f runner/sandbox/Dockerfile .
#   docker run --rm -i --network none --read-only --tmpfs /tmp:size=16m \
#     --cpus 1 --memory 2g --pids-limit 32 --cap-drop ALL --security-opt no-new-privileges \
#     aliensilicon-judge < submission.py
FROM python:3.12-slim
RUN pip install --no-cache-dir "numpy>=1.26,<3" "z3-solver>=4.12" "jsonschema>=4.20"
WORKDIR /opt/aliensilicon
COPY aliensilicon ./aliensilicon
COPY judge ./judge
RUN useradd --uid 10001 --no-create-home judge
USER judge
ENV PYTHONDONTWRITEBYTECODE=1
# Reads DSL source on stdin, prints {"ok": {name: ir_json}} or {"rejected": reason}.
ENTRYPOINT ["python", "-I", "-m", "judge.sandbox_worker"]

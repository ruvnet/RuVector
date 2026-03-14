# Minimal runtime image — binary is pre-built and copied in
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY mcp-brain-server /usr/local/bin/mcp-brain-server

ENV PORT=8080
EXPOSE 8080

CMD ["mcp-brain-server"]

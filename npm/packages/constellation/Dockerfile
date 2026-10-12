FROM node:22-bookworm-slim
WORKDIR /app
COPY --chown=node:node src ./src
USER node
ENTRYPOINT ["node", "src/cloud-run.mjs"]

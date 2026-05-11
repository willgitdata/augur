# Multi-stage Dockerfile for the Augur server.
#
# Stage 1: install deps + build the workspace.
# Stage 2: copy the built artifacts into a slim runtime image.
#
# We deliberately ship only the `@augur-rag/core` and `@augur-rag/server`
# packages — the dashboard runs as a separate service.

FROM node:20-alpine AS builder
WORKDIR /app
RUN corepack enable

COPY pnpm-workspace.yaml package.json tsconfig.base.json .npmrc ./
COPY packages/core/package.json packages/core/tsconfig.json ./packages/core/
COPY packages/server/package.json packages/server/tsconfig.json ./packages/server/
RUN pnpm install --frozen-lockfile=false

COPY packages/core ./packages/core
COPY packages/server ./packages/server
RUN pnpm --filter @augur-rag/core build && pnpm --filter @augur-rag/server build

# ---- Runtime ----
FROM node:20-alpine AS runtime
WORKDIR /app
RUN corepack enable
ENV NODE_ENV=production

COPY --from=builder /app/package.json /app/pnpm-workspace.yaml /app/.npmrc ./
COPY --from=builder /app/packages/core/package.json ./packages/core/
COPY --from=builder /app/packages/core/dist        ./packages/core/dist
COPY --from=builder /app/packages/server/package.json ./packages/server/
COPY --from=builder /app/packages/server/dist        ./packages/server/dist

RUN pnpm install --prod --frozen-lockfile=false

EXPOSE 3001
CMD ["node", "packages/server/dist/cli.js"]

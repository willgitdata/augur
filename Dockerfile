# Multi-stage Dockerfile for the Augur server.
#
# Stage 1: install deps + build the workspace.
# Stage 2: copy the built artifacts into a slim runtime image.
#
# We deliberately ship only the `@augur-rag/core` and `@augur-rag/server`
# packages — the dashboard runs as a separate service.
#
# Hardening choices:
#  - `--frozen-lockfile` (default for pnpm in CI) so a tampered registry
#    mirror or transient resolve quirk can't pull a different version than
#    the one pinned in pnpm-lock.yaml.
#  - Runtime stage drops to the unprivileged `node` user (UID 1000, shipped
#    in the node:alpine base) so a process compromise can't write outside
#    its own working directory.

FROM node:20-alpine AS builder
WORKDIR /app
RUN corepack enable

COPY pnpm-workspace.yaml pnpm-lock.yaml package.json tsconfig.base.json .npmrc ./
COPY packages/core/package.json packages/core/tsconfig.json ./packages/core/
COPY packages/server/package.json packages/server/tsconfig.json ./packages/server/
RUN pnpm install --frozen-lockfile

COPY packages/core ./packages/core
COPY packages/server ./packages/server
RUN pnpm --filter @augur-rag/core build && pnpm --filter @augur-rag/server build

# ---- Runtime ----
FROM node:20-alpine AS runtime
WORKDIR /app
RUN corepack enable
ENV NODE_ENV=production

# Copy with ownership set to `node` so the unprivileged user can read
# everything we put under /app. Avoids a separate `chown -R` layer.
COPY --from=builder --chown=node:node /app/package.json /app/pnpm-workspace.yaml /app/pnpm-lock.yaml /app/.npmrc ./
COPY --from=builder --chown=node:node /app/packages/core/package.json ./packages/core/
COPY --from=builder --chown=node:node /app/packages/core/dist        ./packages/core/dist
COPY --from=builder --chown=node:node /app/packages/server/package.json ./packages/server/
COPY --from=builder --chown=node:node /app/packages/server/dist        ./packages/server/dist

RUN pnpm install --prod --frozen-lockfile && chown -R node:node /app

USER node

EXPOSE 3001
CMD ["node", "packages/server/dist/cli.js"]

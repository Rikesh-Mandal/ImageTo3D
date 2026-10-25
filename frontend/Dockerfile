
# Stage 1: Install dependencies.
FROM node:24-bookworm-slim AS dependencies

WORKDIR /app

COPY package.json package-lock.json ./

RUN npm ci


# Stage 2: Build the Next.js application.
FROM node:24-bookworm-slim AS builder

WORKDIR /app

ENV NEXT_TELEMETRY_DISABLED=1

# The backend service name is resolved within Docker.
ARG FASTAPI_URL=http://backend:8001
ENV FASTAPI_URL=${FASTAPI_URL}

COPY --from=dependencies /app/node_modules ./node_modules

COPY . .

RUN npm run build


# Stage 3: Small production runtime.
FROM node:24-bookworm-slim AS runner

WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

RUN mkdir -p .next \
    && chown node:node .next

# Copy the standalone production server.
COPY --from=builder --chown=node:node \
    /app/.next/standalone ./

# Copy browser assets.
COPY --from=builder --chown=node:node \
    /app/.next/static ./.next/static

# Copy files served from the public directory.
COPY --from=builder --chown=node:node \
    /app/public ./public

USER node

EXPOSE 3000

CMD ["node", "server.js"]

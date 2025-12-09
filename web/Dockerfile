# Dockerfile for UIT Chatbot Frontend
#
# Build the React app and serve it with a lightweight HTTP server.

FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm ci

# Copy source files
COPY . .

# Build the app
RUN npm run build

# Production stage: serve the built app
FROM node:20-alpine

WORKDIR /app

# Install serve globally
RUN npm install -g serve

# Copy built files from builder stage
COPY --from=builder /app/dist ./dist

# Expose port 4173 (default for serve)
EXPOSE 4173

# Serve the built app
# Note: VITE_API_BASE_URL should be set at runtime if different from default
CMD ["serve", "-s", "dist", "-l", "4173"]


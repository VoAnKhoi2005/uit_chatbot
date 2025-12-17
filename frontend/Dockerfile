# Dockerfile for UIT Chatbot Frontend
#
# Build the React app and serve it with a lightweight HTTP server.

FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm install

# Copy source files
COPY . .

# Copy PDF.js worker to public directory (use react-pdf's bundled version to match API version)
RUN cp node_modules/react-pdf/node_modules/pdfjs-dist/build/pdf.worker.min.mjs public/pdf.worker.min.mjs || \
    cp node_modules/pdfjs-dist/build/pdf.worker.min.mjs public/pdf.worker.min.mjs

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

# Create entrypoint script to inject runtime config
RUN echo '#!/bin/sh' > /entrypoint.sh && \
    echo 'if [ -n "$API_BASE_URL" ]; then' >> /entrypoint.sh && \
    echo '  sed -i "s|__API_BASE_URL__|$API_BASE_URL|g" /app/dist/config.js' >> /entrypoint.sh && \
    echo 'fi' >> /entrypoint.sh && \
    echo 'exec serve -s dist -l 4173' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]



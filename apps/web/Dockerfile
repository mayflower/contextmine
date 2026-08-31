FROM node:20-alpine

WORKDIR /app

# Create the runtime user before dependency installation so the image does not
# need to recursively chown node_modules after every source change.
RUN adduser --disabled-password --gecos "" appuser && \
    chown appuser:appuser /app

# Copy package files
COPY --chown=appuser:appuser package*.json ./

USER appuser

# Install dependencies
RUN npm ci

# Copy source files
COPY --chown=appuser:appuser . .

# Expose port
EXPOSE 5173

# Run dev server (for development)
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

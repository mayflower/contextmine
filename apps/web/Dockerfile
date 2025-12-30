FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source files
COPY . .

# Create non-root user and fix ownership
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5173

# Run dev server (for development)
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

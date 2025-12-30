"""Rate limiting configuration using slowapi."""

from slowapi import Limiter
from slowapi.util import get_remote_address

# Create limiter with IP-based rate limiting
limiter = Limiter(key_func=get_remote_address)

# Rate limit constants
RATE_LIMIT_DEFAULT = "60/minute"  # Default rate limit for most endpoints
RATE_LIMIT_AUTH = "10/minute"  # Stricter limit for auth endpoints
RATE_LIMIT_SEARCH = "30/minute"  # Limit for search/context endpoints
RATE_LIMIT_RESEARCH = "10/minute"  # Limit for expensive research operations

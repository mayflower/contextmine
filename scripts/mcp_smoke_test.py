#!/usr/bin/env python3
"""MCP endpoint smoke test script.

This script tests the MCP endpoint by:
1. Calling the context.get_markdown tool via HTTP
2. Verifying the response contains a Sources section

Usage:
    python scripts/mcp_smoke_test.py --token YOUR_MCP_TOKEN [--base-url http://localhost:8000]
"""

import argparse
import json
import sys

import requests


def test_mcp_endpoint(base_url: str, token: str, query: str = "How do I use the API?") -> bool:
    """Test the MCP endpoint with a context query.

    Args:
        base_url: Base URL of the API (e.g., http://localhost:8000)
        token: MCP Bearer token
        query: Query to test with

    Returns:
        True if test passes, False otherwise
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    mcp_url = f"{base_url}/mcp"

    print(f"Testing MCP endpoint at {mcp_url}")
    print(f"Query: {query}")
    print("-" * 50)

    # Call the context.get_markdown tool via JSON-RPC
    request_body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "context.get_markdown",
            "arguments": {
                "query": query,
            },
        },
    }

    print("Sending tool call request...")

    try:
        response = requests.post(mcp_url, headers=headers, json=request_body, timeout=60)
    except requests.RequestException as e:
        print(f"FAIL: Request error: {e}")
        return False

    if response.status_code == 401:
        print("FAIL: Authentication failed - check your token")
        return False

    if response.status_code == 403:
        print("FAIL: Origin not allowed - check MCP_ALLOWED_ORIGINS")
        return False

    if response.status_code != 200:
        print(f"FAIL: MCP endpoint returned {response.status_code}")
        print(f"Response: {response.text}")
        return False

    print(f"MCP endpoint returned {response.status_code}")

    # Parse response
    try:
        data = response.json()
    except json.JSONDecodeError as e:
        print(f"FAIL: Invalid JSON response: {e}")
        print(f"Response text: {response.text[:500]}")
        return False

    # Extract result text
    result_text = None
    if "result" in data:
        result = data["result"]
        if isinstance(result, dict) and "content" in result:
            for content in result["content"]:
                if content.get("type") == "text":
                    result_text = content.get("text", "")
                    break
        elif isinstance(result, list):
            for item in result:
                if isinstance(item, dict) and item.get("type") == "text":
                    result_text = item.get("text", "")
                    break

    if not result_text:
        print("FAIL: No result text in response")
        print(f"Response: {json.dumps(data, indent=2)[:500]}")
        return False

    print("-" * 50)
    print("Response (first 500 chars):")
    print(result_text[:500])
    print("-" * 50)

    # Verify the response contains expected content
    if "Sources" not in result_text and "## Sources" not in result_text:
        print("FAIL: Response does not contain 'Sources' section")
        return False

    print("PASS: Response contains 'Sources' section")
    return True


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MCP endpoint smoke test")
    parser.add_argument("--token", required=True, help="MCP Bearer token")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--query", default="How do I use the API?", help="Test query")
    args = parser.parse_args()

    success = test_mcp_endpoint(args.base_url, args.token, args.query)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

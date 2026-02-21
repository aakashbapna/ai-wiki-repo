"""Simple GET API smoke test."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from typing import Any


BASE_URL = "http://127.0.0.1:5000/api"


def http_get_json(url: str) -> tuple[int, Any]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = int(resp.status)
            body = resp.read().decode("utf-8")
            return status, json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        payload = json.loads(body) if body else {}
        return int(exc.code), payload


def assert_status(status: int, expected: int, url: str) -> None:
    if status != expected:
        raise AssertionError(f"GET {url} expected {expected}, got {status}")


def main() -> int:
    print("Testing GET endpoints:")
    status, payload = http_get_json(f"{BASE_URL}/repos")
    assert_status(status, 200, f"{BASE_URL}/repos")
    print(f"OK: {BASE_URL}/repos")

    repos = payload.get("repos", [])
    if not isinstance(repos, list) or not repos:
        print("No repos available to test detail endpoints.")
        return 0

    repo_hash = str(repos[0].get("repo_hash") or "")
    if not repo_hash:
        raise AssertionError("repo_hash missing in repos response.")

    status, _ = http_get_json(f"{BASE_URL}/repos/{repo_hash}")
    assert_status(status, 200, f"{BASE_URL}/repos/{repo_hash}")
    print(f"OK: {BASE_URL}/repos/{repo_hash}")

    status, _ = http_get_json(f"{BASE_URL}/repos/{repo_hash}/files")
    assert_status(status, 200, f"{BASE_URL}/repos/{repo_hash}/files")
    print(f"OK: {BASE_URL}/repos/{repo_hash}/files")

    status, _ = http_get_json(f"{BASE_URL}/repos/{repo_hash}/subsystems")
    assert_status(status, 200, f"{BASE_URL}/repos/{repo_hash}/subsystems")
    print(f"OK: {BASE_URL}/repos/{repo_hash}/subsystems")

    status, _ = http_get_json(f"{BASE_URL}/repos/{repo_hash}/index")
    if status not in {200, 404}:
        raise AssertionError(f"GET {BASE_URL}/repos/{repo_hash}/index unexpected {status}")
    print(f"OK: {BASE_URL}/repos/{repo_hash}/index ({status})")

    status, _ = http_get_json(f"{BASE_URL}/openai/health")
    assert_status(status, 200, f"{BASE_URL}/openai/health")
    print(f"OK: {BASE_URL}/openai/health")

    print("API GET smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

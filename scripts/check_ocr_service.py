#!/usr/bin/env python3
import argparse
import json
import sys
import urllib.error
import urllib.request


def fetch_json(url: str, timeout: int):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, json.loads(body)


def main():
    parser = argparse.ArgumentParser(description="Check OCR service health/info/stats endpoints.")
    parser.add_argument("--host", default="127.0.0.1", help="OCR service host")
    parser.add_argument("--port", type=int, default=9035, help="OCR service port")
    parser.add_argument("--timeout", type=int, default=10, help="HTTP timeout in seconds")
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    endpoints = ["/health", "/info", "/stats"]
    failed = False

    for endpoint in endpoints:
        url = base + endpoint
        try:
            status, payload = fetch_json(url, args.timeout)
            print(f"[OK] {endpoint} status={status}")
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        except urllib.error.HTTPError as exc:
            failed = True
            print(f"[FAIL] {endpoint} http_status={exc.code}", file=sys.stderr)
        except Exception as exc:
            failed = True
            print(f"[FAIL] {endpoint} error={exc}", file=sys.stderr)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

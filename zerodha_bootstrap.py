import argparse
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from typing import Any

from dotenv import load_dotenv, set_key

from src.zerodha_client import ZerodhaClient
from src.zerodha_rest_client import ZerodhaRestClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zerodha connect helper")
    parser.add_argument("--mode", choices=["rest", "sdk"], default="rest", help="Client mode: direct REST handshake or Kite SDK")
    parser.add_argument("--show-login-url", action="store_true", help="Print the Kite login URL")
    parser.add_argument("--request-token", default="", help="Request token from Zerodha redirect URL")
    parser.add_argument("--redirect-params", default="", help="Optional redirect_params appended to login URL")
    parser.add_argument("--auto-login", action="store_true", help="Open browser and auto-capture request_token on local redirect URL")
    parser.add_argument("--redirect-url", default="", help="Local redirect URL set in Kite app, e.g. http://127.0.0.1:8765/callback")
    parser.add_argument("--timeout-seconds", type=int, default=300, help="Timeout while waiting for redirect callback")
    parser.add_argument("--no-open-browser", action="store_true", help="Do not auto-open browser for --auto-login")
    parser.add_argument("--persist-token", action="store_true", help="Persist generated access token into .env")
    parser.add_argument("--profile", action="store_true", help="Fetch and print account profile")
    parser.add_argument("--margins", action="store_true", help="Fetch and print account margins")
    parser.add_argument("--margins-segment", default="", help="Optional margin segment: equity or commodity")
    parser.add_argument("--logout", action="store_true", help="Invalidate current access token")
    parser.add_argument(
        "--ltp",
        nargs="*",
        default=[],
        help="Market symbols for LTP, e.g. NSE:RELIANCE",
    )
    parser.add_argument(
        "--save-session",
        action="store_true",
        help="Save generated token metadata to logs/zerodha_session.json",
    )
    return parser.parse_args()


def capture_request_token(redirect_url: str, timeout_seconds: int) -> str:
    parsed = urlparse(redirect_url)
    host = parsed.hostname
    port = parsed.port
    path = parsed.path or "/"

    if host not in {"127.0.0.1", "localhost"}:
        raise RuntimeError("--auto-login requires localhost redirect URL (127.0.0.1 or localhost)")
    if port is None:
        raise RuntimeError("Redirect URL must include a port, e.g. http://127.0.0.1:8765/callback")

    result: dict[str, str] = {"request_token": "", "error": ""}

    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            req = urlparse(self.path)
            if req.path != path:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Invalid callback path")
                return

            qs = parse_qs(req.query)
            token = (qs.get("request_token") or [""])[0]
            status = (qs.get("status") or [""])[0]

            if token:
                result["request_token"] = token
                body = b"Kite login successful. You can close this window."
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if status:
                result["error"] = f"Kite callback status={status}"
            else:
                result["error"] = "Kite callback missing request_token"

            body = f"Login callback error: {result['error']}".encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    server = HTTPServer((host, port), CallbackHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    started = time.time()
    try:
        while True:
            if result["request_token"]:
                return result["request_token"]
            if result["error"]:
                raise RuntimeError(result["error"])
            if time.time() - started > timeout_seconds:
                raise TimeoutError(f"Timed out waiting for callback after {timeout_seconds} seconds")
            time.sleep(0.25)
    finally:
        server.shutdown()
        server.server_close()


def persist_access_token(token: str) -> Path:
    env_path = Path(".env")
    if not env_path.exists():
        env_path.write_text("", encoding="utf-8")
    set_key(str(env_path), "ZERODHA_ACCESS_TOKEN", token)
    return env_path


def main() -> None:
    load_dotenv()
    args = parse_args()
    client: Any = ZerodhaRestClient.from_env() if args.mode == "rest" else ZerodhaClient.from_env()

    request_token = args.request_token.strip()

    if args.auto_login:
        redirect_url = (args.redirect_url or "").strip()
        if not redirect_url:
            raise RuntimeError("--auto-login requires --redirect-url set to your localhost callback URL")

        login_url = client.login_url(args.redirect_params.strip()) if args.mode == "rest" else client.login_url()
        print("Starting local callback server...")
        print(f"Redirect URL (must match Kite app): {redirect_url}")
        print(f"Timeout: {args.timeout_seconds} seconds")
        print("Login URL:")
        print(login_url)

        if not args.no_open_browser:
            webbrowser.open(login_url)

        request_token = capture_request_token(redirect_url, timeout_seconds=max(30, args.timeout_seconds))
        print("Captured request_token from callback.")

    if args.show_login_url:
        print("Open this URL and complete login:")
        print(client.login_url(args.redirect_params.strip()) if args.mode == "rest" else client.login_url())

    if request_token:
        token = client.generate_access_token(request_token)
        print("Generated access token. Set this in ZERODHA_ACCESS_TOKEN:")
        print(token)

        if args.persist_token:
            env_path = persist_access_token(token)
            print(f"Persisted access token to {env_path}")

        if args.save_session:
            out_path = Path("logs") / "zerodha_session.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps({"access_token": token}, indent=2),
                encoding="utf-8",
            )
            print(f"Saved session token file: {out_path}")

    if args.profile:
        print(json.dumps(client.profile(), indent=2))

    if args.margins:
        if args.mode == "rest":
            print(json.dumps(client.margins(args.margins_segment.strip()), indent=2))
        else:
            print(json.dumps(client.margins(), indent=2))

    if args.ltp:
        print(json.dumps(client.ltp(args.ltp), indent=2))

    if args.logout:
        if args.mode != "rest":
            raise RuntimeError("--logout is currently supported only in --mode rest")
        print(json.dumps({"logout": client.logout()}, indent=2))

    if not any([
        args.show_login_url,
        args.request_token,
        args.auto_login,
        args.profile,
        args.margins,
        args.ltp,
        args.logout,
    ]):
        print("No action selected. Example:")
        print("python zerodha_bootstrap.py --mode rest --show-login-url")
        print("python zerodha_bootstrap.py --mode rest --auto-login --redirect-url http://127.0.0.1:8765/callback --persist-token")


if __name__ == "__main__":
    main()

import os
from dataclasses import dataclass
from typing import Any

from kiteconnect import KiteConnect


@dataclass
class ZerodhaConfig:
    api_key: str
    api_secret: str
    access_token: str


class ZerodhaClient:
    def __init__(self, config: ZerodhaConfig) -> None:
        self.config = config
        self.kite = KiteConnect(api_key=config.api_key)
        if config.access_token:
            self.kite.set_access_token(config.access_token)

    @classmethod
    def from_env(cls) -> "ZerodhaClient":
        api_key = os.getenv("ZERODHA_API_KEY", "").strip()
        api_secret = os.getenv("ZERODHA_API_SECRET", "").strip()
        access_token = os.getenv("ZERODHA_ACCESS_TOKEN", "").strip()

        if not api_key:
            raise RuntimeError("Missing ZERODHA_API_KEY in environment")

        return cls(
            ZerodhaConfig(
                api_key=api_key,
                api_secret=api_secret,
                access_token=access_token,
            )
        )

    def login_url(self) -> str:
        return self.kite.login_url()

    def generate_access_token(self, request_token: str) -> str:
        if not self.config.api_secret:
            raise RuntimeError("Missing ZERODHA_API_SECRET in environment")

        session = self.kite.generate_session(
            request_token=request_token,
            api_secret=self.config.api_secret,
        )
        token = session["access_token"]
        self.kite.set_access_token(token)
        return token

    def profile(self) -> dict[str, Any]:
        return self.kite.profile()

    def margins(self) -> dict[str, Any]:
        return self.kite.margins()

    def ltp(self, symbols: list[str]) -> dict[str, Any]:
        return self.kite.ltp(symbols)

import hashlib
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote_plus

import requests


@dataclass
class ZerodhaRestConfig:
    api_key: str
    api_secret: str
    access_token: str
    root_url: str = "https://api.kite.trade"
    login_url_root: str = "https://kite.zerodha.com/connect/login"


class ZerodhaRestClient:
    kite_header_version = "3"

    def __init__(self, config: ZerodhaRestConfig) -> None:
        self.config = config
        self.access_token = config.access_token
        self.session = requests.Session()

    @classmethod
    def from_env(cls) -> "ZerodhaRestClient":
        api_key = os.getenv("ZERODHA_API_KEY", "").strip()
        api_secret = os.getenv("ZERODHA_API_SECRET", "").strip()
        access_token = os.getenv("ZERODHA_ACCESS_TOKEN", "").strip()

        if not api_key:
            raise RuntimeError("Missing ZERODHA_API_KEY in environment")

        return cls(
            ZerodhaRestConfig(
                api_key=api_key,
                api_secret=api_secret,
                access_token=access_token,
            )
        )

    def login_url(self, redirect_params: str = "") -> str:
        url = f"{self.config.login_url_root}?v=3&api_key={self.config.api_key}"
        if redirect_params:
            url = f"{url}&redirect_params={quote_plus(redirect_params)}"
        return url

    def _headers(self, with_auth: bool = True) -> dict[str, str]:
        headers = {
            "X-Kite-Version": self.kite_header_version,
            "User-Agent": "quantpulse-zerodha-rest",
        }
        if with_auth and self.access_token:
            headers["Authorization"] = f"token {self.config.api_key}:{self.access_token}"
        return headers

    def _json(self, response: requests.Response) -> Any:
        payload = response.json()
        if payload.get("status") == "error" or payload.get("error_type"):
            msg = payload.get("message", "Unknown Kite API error")
            raise RuntimeError(msg)
        return payload.get("data")

    def generate_access_token(self, request_token: str) -> str:
        if not self.config.api_secret:
            raise RuntimeError("Missing ZERODHA_API_SECRET in environment")

        checksum = hashlib.sha256(
            (self.config.api_key + request_token + self.config.api_secret).encode("utf-8")
        ).hexdigest()

        response = self.session.post(
            f"{self.config.root_url}/session/token",
            headers=self._headers(with_auth=False),
            data={
                "api_key": self.config.api_key,
                "request_token": request_token,
                "checksum": checksum,
            },
            timeout=15,
        )
        response.raise_for_status()

        data = self._json(response)
        token = data["access_token"]
        self.access_token = token
        return token

    def profile(self) -> dict[str, Any]:
        response = self.session.get(
            f"{self.config.root_url}/user/profile",
            headers=self._headers(with_auth=True),
            timeout=15,
        )
        response.raise_for_status()
        return self._json(response)

    def margins(self, segment: str = "") -> dict[str, Any]:
        endpoint = f"/user/margins/{segment.strip()}" if segment else "/user/margins"
        response = self.session.get(
            f"{self.config.root_url}{endpoint}",
            headers=self._headers(with_auth=True),
            timeout=15,
        )
        response.raise_for_status()
        return self._json(response)

    def ltp(self, symbols: list[str]) -> dict[str, Any]:
        if not symbols:
            return {}
        response = self.session.get(
            f"{self.config.root_url}/quote/ltp",
            headers=self._headers(with_auth=True),
            params={"i": symbols},
            timeout=15,
        )
        response.raise_for_status()
        return self._json(response)

    def quote_ohlc(self, symbols: list[str]) -> dict[str, Any]:
        if not symbols:
            return {}
        response = self.session.get(
            f"{self.config.root_url}/quote/ohlc",
            headers=self._headers(with_auth=True),
            params={"i": symbols},
            timeout=15,
        )
        response.raise_for_status()
        return self._json(response)

    def logout(self) -> bool:
        if not self.access_token:
            return True
        response = self.session.delete(
            f"{self.config.root_url}/session/token",
            headers=self._headers(with_auth=True),
            params={
                "api_key": self.config.api_key,
                "access_token": self.access_token,
            },
            timeout=15,
        )
        response.raise_for_status()
        result = self._json(response)
        if result is True:
            self.access_token = ""
        return bool(result)

    def instruments(self, exchange: str = None) -> list[dict]:
        """
        Fetch the instruments CSV dump (gzipped).
        Returns list of dicts with: instrument_token, exchange_token, tradingsymbol, name, etc.
        
        Args:
            exchange: Optional exchange filter ("NSE", "BSE", "MCX", "NCDEX")
        """
        if exchange:
            url = f"{self.config.root_url}/instruments/{exchange}"
        else:
            url = f"{self.config.root_url}/instruments"
        
        headers = self._headers(with_auth=True)
        response = self.session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Response is gzipped CSV; parse it
        import io
        import csv
        import gzip
        
        content = response.content
        if content.startswith(b'\x1f\x8b'):  # gzip magic bytes
            content = gzip.decompress(content)
        
        reader = csv.DictReader(io.StringIO(content.decode('utf-8')))
        return list(reader)

    def get_instrument_token(self, exchange: str, tradingsymbol: str, cache: dict = None) -> str:
        """
        Get instrument_token for a given exchange:tradingsymbol.
        Uses cache if provided to avoid repeated API calls.
        
        Args:
            exchange: "NSE" or "BSE"
            tradingsymbol: Stock symbol like "INFY"
            cache: Optional dict to cache results {(exchange, symbol) -> token}
        
        Returns:
            instrument_token as string
        """
        if cache is None:
            cache = {}
        
        key = (exchange, tradingsymbol)
        if key in cache:
            return cache[key]
        
        instruments = self.instruments(exchange=exchange)
        for instr in instruments:
            if instr.get('tradingsymbol') == tradingsymbol:
                token = instr.get('instrument_token')
                cache[key] = token
                return token
        
        raise ValueError(f"Instrument not found: {exchange}:{tradingsymbol}")

    def historical_candles(
        self,
        instrument_token: str,
        interval: str,
        from_date: str,
        to_date: str,
        continuous: int = 0,
        oi: int = 0
    ) -> list[list]:
        """
        Fetch historical OHLCV candles for an instrument.
        
        Args:
            instrument_token: Token from instruments API
            interval: "day", "minute", "5minute", etc.
            from_date: "yyyy-mm-dd hh:mm:ss" or "yyyy-mm-dd"
            to_date: "yyyy-mm-dd hh:mm:ss" or "yyyy-mm-dd"
            continuous: 1 for continuous contracts, 0 otherwise
            oi: 1 to include open interest, 0 otherwise
        
        Returns:
            List of [timestamp, open, high, low, close, volume, oi?]
        """
        url = f"{self.config.root_url}/instruments/historical/{instrument_token}/{interval}"
        headers = self._headers(with_auth=True)
        
        params = {
            "from": from_date,
            "to": to_date,
            "continuous": continuous,
            "oi": oi
        }
        
        response = self.session.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        return data.get("data", {}).get("candles", [])

import os
import requests
from typing import Any, Dict, List, Optional

TRADIER_BASE_URL = "https://api.tradier.com/v1"

class TradierClient:
    def __init__(self, token: str, account_mode: str = "prod"):
        """
        token: Tradier bearer token
        account_mode: 'prod' or 'sandbox' (Tradier also has sandbox URLs, but many users use prod)
        """
        if not token:
            raise ValueError("Tradier token is required.")
        self.token = token

        # If you use Tradier sandbox, you may need a different base URL.
        # Keep this simple; adjust if your account uses sandbox.
        self.base_url = TRADIER_BASE_URL

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/json",
            }
        )

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        r = self.session.get(url, params=params, timeout=20)
        r.raise_for_status()
        return r.json()

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        data = self._get("/markets/quotes", params={"symbols": symbol, "greeks": "false"})
        # Tradier wraps responses under quote/quotes depending on cardinality.
        quotes = data.get("quotes", {}).get("quote")
        if isinstance(quotes, list):
            return quotes[0]
        return quotes

    def get_expirations(self, symbol: str, include_all_roots: bool = False) -> List[str]:
        params = {"symbol": symbol, "includeAllRoots": "true" if include_all_roots else "false"}
        data = self._get("/markets/options/expirations", params=params)
        dates = data.get("expirations", {}).get("date", [])
        return dates if isinstance(dates, list) else [dates]

    def get_chain(self, symbol: str, expiration: str, greeks: bool = True) -> List[Dict[str, Any]]:
        params = {
            "symbol": symbol,
            "expiration": expiration,
            "greeks": "true" if greeks else "false",
        }
        data = self._get("/markets/options/chains", params=params)
        opts = data.get("options", {}).get("option", [])
        return opts if isinstance(opts, list) else [opts]

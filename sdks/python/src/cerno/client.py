"""Sync and async clients."""

from __future__ import annotations

from typing import Any

import httpx

from ._builder import SystemOneBuilder
from ._models import Answers, ApiError, TransportError, UnexpectedResponse

DEFAULT_TIMEOUT = 60.0


def _decode(response: httpx.Response) -> Any:
    """Turn a response into data, or into the most specific error we can justify."""
    if response.is_success:
        try:
            return response.json()
        except ValueError:
            raise UnexpectedResponse(response.status_code, response.text) from None

    try:
        body = response.json()
        code = body["code"]
    except Exception:
        # Not our error shape, so do not pretend to know what went wrong.
        raise UnexpectedResponse(response.status_code, response.text) from None

    raise ApiError(
        status=response.status_code,
        code=code,
        message=body.get("message", ""),
        question_id=body.get("question_id"),
    )


def _answers(response: httpx.Response) -> Answers:
    """Read a response as answers. A 2xx body in some other shape is not ours to interpret."""
    data = _decode(response)
    try:
        return Answers(data)
    except (KeyError, TypeError, AttributeError):
        raise UnexpectedResponse(response.status_code, response.text) from None


class Client:
    """Synchronous client.

    >>> client = Client("http://localhost:3000")                     # doctest: +SKIP
    >>> answers = (client.systemone("Ticket: server room at 31C.")   # doctest: +SKIP
    ...     .noul("urgent", "Is this urgent?")
    ...     .choice("team", "Which team?", ["IT", "Facility"])
    ...     .send())
    """

    def __init__(
        self,
        base_url: str = "http://localhost:3000",
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._http = client or httpx.Client(timeout=timeout)
        self._owns_http = client is None

    def systemone(self, state: str) -> SystemOneBuilder:
        """Start a request about ``state``."""
        return SystemOneBuilder(state, self._send)

    def _send(self, body: dict[str, Any]) -> Answers:
        try:
            response = self._http.post(f"{self.base_url}/v1/systemone", json=body)
        except httpx.HTTPError as error:
            raise TransportError(f"could not reach cerno: {error}") from error
        return _answers(response)

    def models(self) -> dict[str, Any]:
        """The models this service will answer for."""
        try:
            response = self._http.get(f"{self.base_url}/v1/models")
        except httpx.HTTPError as error:
            raise TransportError(f"could not reach cerno: {error}") from error
        return _decode(response)

    def health(self) -> bool:
        try:
            return self._http.get(f"{self.base_url}/health").is_success
        except httpx.HTTPError:
            return False

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class AsyncClient:
    """Asynchronous client. Same surface as :class:`Client`, with ``await`` on ``send()``."""

    def __init__(
        self,
        base_url: str = "http://localhost:3000",
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._http = client or httpx.AsyncClient(timeout=timeout)
        self._owns_http = client is None

    def systemone(self, state: str) -> SystemOneBuilder:
        return SystemOneBuilder(state, self._send)

    async def _send(self, body: dict[str, Any]) -> Answers:
        try:
            response = await self._http.post(f"{self.base_url}/v1/systemone", json=body)
        except httpx.HTTPError as error:
            raise TransportError(f"could not reach cerno: {error}") from error
        return _answers(response)

    async def models(self) -> dict[str, Any]:
        try:
            response = await self._http.get(f"{self.base_url}/v1/models")
        except httpx.HTTPError as error:
            raise TransportError(f"could not reach cerno: {error}") from error
        return _decode(response)

    async def health(self) -> bool:
        try:
            return (await self._http.get(f"{self.base_url}/health")).is_success
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> "AsyncClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

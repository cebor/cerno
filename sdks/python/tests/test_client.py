"""Failure modes outside the shared conformance cases: transport and response shape.

Everything this package raises for a failed request is a ``CernoError``, so a caller can catch
one type and know nothing slipped past it.
"""

import httpx
import pytest

from cerno import AsyncClient, CernoError, Client, TransportError, UnexpectedResponse


def client_with(handler) -> Client:
    return Client("http://cerno.test", client=httpx.Client(transport=httpx.MockTransport(handler)))


def refuse(request):
    raise httpx.ConnectError("connection refused", request=request)


def test_an_unreachable_service_is_a_transport_error():
    with pytest.raises(TransportError) as caught:
        client_with(refuse).systemone("state").noul("q", "Urgent?").send()

    assert isinstance(caught.value, CernoError)
    assert isinstance(caught.value.__cause__, httpx.ConnectError)


def test_models_on_an_unreachable_service_is_a_transport_error():
    with pytest.raises(TransportError):
        client_with(refuse).models()


async def test_the_async_client_raises_a_transport_error_too():
    client = AsyncClient(
        "http://cerno.test", client=httpx.AsyncClient(transport=httpx.MockTransport(refuse))
    )

    with pytest.raises(TransportError):
        await client.systemone("state").noul("q", "Urgent?").send()
    await client.aclose()


def test_a_success_status_with_a_body_that_is_not_json_is_unexpected():
    client = client_with(lambda _r: httpx.Response(200, text="<html>hello</html>"))

    with pytest.raises(UnexpectedResponse) as caught:
        client.systemone("state").noul("q", "Urgent?").send()

    assert caught.value.status == 200


def test_a_success_status_with_json_in_another_shape_is_unexpected():
    client = client_with(lambda _r: httpx.Response(200, json={"hello": "world"}))

    with pytest.raises(UnexpectedResponse):
        client.systemone("state").noul("q", "Urgent?").send()


def test_health_is_false_when_the_service_cannot_be_reached():
    assert client_with(refuse).health() is False

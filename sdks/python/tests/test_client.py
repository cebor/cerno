"""Failure modes outside the shared conformance cases: transport, response shape, builder misuse.

Everything this package raises for a failed request is a ``CernoError``, so a caller can catch
one type and know nothing slipped past it.
"""

import httpx
import pytest

from cerno import AsyncClient, CernoError, Client, TransportError


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


def test_health_is_false_when_the_service_cannot_be_reached():
    assert client_with(refuse).health() is False


def builder():
    return Client("http://unused.invalid").systemone("state")


def test_a_choice_without_options_is_refused_rather_than_split_into_characters():
    with pytest.raises(TypeError, match="list of options"):
        builder().choice("team", "Which team?")


def test_a_string_passed_as_options_is_refused():
    with pytest.raises(TypeError):
        builder().choice("team", "Which team?", "IT, Facility")


@pytest.mark.parametrize("levels", ["low, high", True])
def test_a_score_rubric_must_be_a_count_or_a_list(levels):
    with pytest.raises(TypeError):
        builder().score("sev", "How bad?", levels)


def test_the_valid_shapes_still_build():
    body = (
        builder()
        .choice("team", "Which team?", ("IT", "Facility"))
        .choice("mood", ["up", "down"])
        .score("sev", "How bad?", 5)
        .score("stars", "How many?", ["low", "high"])
        .body()
    )

    assert body["questions"][0]["choice"]["options"] == ["IT", "Facility"]
    assert body["questions"][1]["choice"] == {"options": ["up", "down"]}
    assert body["questions"][2]["score"]["levels"] == 5
    assert body["questions"][3]["score"]["levels"] == ["low", "high"]

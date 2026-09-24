"""Runs the shared conformance cases from spec/conformance/cases.json.

Every cerno SDK runs these same cases, so if the three clients ever disagree about the wire
format, they disagree here first.
"""

import httpx
import pytest

from cerno import ApiError, AsyncClient, Client, MissingAnswer, UnexpectedResponse, WrongAnswerType


def build(client, case):
    """Replay one request case through the public builder."""
    builder = client.systemone(case["state"])

    if "model" in case:
        builder = builder.model(case["model"])
    if "calibration" in case:
        builder = builder.calibration(case["calibration"])

    for question in case["questions"]:
        qid, kind, text = question["id"], question["kind"], question.get("question")

        if kind == "noul":
            builder = builder.noul(qid, text)
        elif kind == "choice":
            options = question["options"]
            # A choice whose options speak for themselves is called without a question.
            builder = builder.choice(qid, options) if text is None else builder.choice(qid, text, options)
        elif kind == "score":
            builder = builder.score(qid, text, question["levels"])
        else:
            raise AssertionError(f"unknown kind {kind}")

    return builder


def transport(status: int, body):
    """An httpx transport that answers every request with one canned response."""
    return httpx.MockTransport(lambda _request: httpx.Response(status, json=body))


def client_for(status: int, body) -> Client:
    return Client("http://cerno.test", client=httpx.Client(transport=transport(status, body)))


def test_request_cases_produce_the_expected_body(cases):
    client = Client("http://unused.invalid")

    for case in cases["requests"]:
        assert build(client, case).body() == case["expect_body"], case["name"]


def test_response_cases_read_back_the_expected_values(cases):
    for case in cases["responses"]:
        name = case["name"]
        answers = (
            client_for(200, case["body"]).systemone("state").noul("ignored", "q").send()
        )
        expect = case["expect"]

        if "model" in expect:
            assert answers.model == expect["model"], name
        if "usage" in expect:
            assert answers.usage.input_tokens == expect["usage"]["input_tokens"], name

        for qid, want in expect["answers"].items():
            kind = want["type"]

            if kind == "noul" and "noul" in want:
                assert answers.noul(qid) == want["noul"], f"{name}/{qid}"
            elif kind == "choice":
                assert answers.choice(qid) == want["choice"], f"{name}/{qid}"
                assert answers.index(qid) == want["index"], f"{name}/{qid}"
            elif kind == "score":
                assert answers.score(qid) == want["score"], f"{name}/{qid}"
                if "legend" in want:
                    assert answers.legend(qid) == want["legend"], f"{name}/{qid}"

            if "truncated" in want:
                assert answers.truncated(qid) is want["truncated"], f"{name}/{qid}"
            if "label_mass" in want:
                assert answers.label_mass(qid) == want["label_mass"], f"{name}/{qid}"
            if "truncated_labels" in want:
                assert answers.truncated_labels(qid) == tuple(want["truncated_labels"]), (
                    f"{name}/{qid}"
                )


def test_error_cases_raise_api_error_with_the_code(cases):
    for case in cases["errors"]:
        name = case["name"]
        client = client_for(case["status"], case["body"])

        with pytest.raises(ApiError) as caught:
            client.systemone("state").noul("q", "Urgent?").send()

        assert caught.value.status == case["status"], name
        assert caught.value.code == case["body"]["code"], name
        assert caught.value.question_id == case["body"].get("question_id"), name


def test_unknown_code_cases_keep_their_code_and_message(cases):
    """A newer service can send a code this client has never heard of. That is still the
    service speaking, so it is an ApiError carrying the code as it came."""
    for case in cases["unknown_codes"]:
        name, body = case["name"], case["body"]
        client = client_for(case["status"], body)

        with pytest.raises(ApiError) as caught:
            client.systemone("state").noul("q", "Urgent?").send()

        assert caught.value.status == case["status"], name
        assert caught.value.code == body["code"], name
        assert caught.value.message == body["message"], name
        assert caught.value.question_id == body.get("question_id"), name


def test_unexpected_cases_are_reported_as_unexpected(cases):
    """A body that is not cerno's - a gateway's HTML page, some other JSON, a 2xx that is not an
    answer - is reported as exactly that, never as a cerno error code it does not carry."""
    for case in cases["unexpected"]:
        name, status, text = case["name"], case["status"], case["body_text"]
        transport = httpx.MockTransport(
            lambda _request, status=status, text=text: httpx.Response(status, text=text)
        )
        client = Client("http://cerno.test", client=httpx.Client(transport=transport))

        with pytest.raises(UnexpectedResponse) as caught:
            client.systemone("state").noul("q", "Urgent?").send()

        assert caught.value.status == status, name


def test_reading_an_answer_as_the_wrong_type_names_both_types(cases):
    answers = (
        client_for(200, cases["responses"][0]["body"])
        .systemone("state")
        .noul("urgent", "Urgent?")
        .send()
    )

    with pytest.raises(WrongAnswerType) as caught:
        answers.choice("urgent")

    assert caught.value.expected == "choice"
    assert caught.value.actual == "noul"

    # A noul carries no confidence, as in JEV, so asking for one is the same mistake.
    with pytest.raises(WrongAnswerType) as caught:
        answers.confidence("urgent")

    assert caught.value.expected == "choice or score"

    with pytest.raises(MissingAnswer):
        answers.noul("nope")


def test_answers_behave_like_a_mapping(cases):
    answers = (
        client_for(200, cases["responses"][0]["body"])
        .systemone("state")
        .noul("urgent", "Urgent?")
        .send()
    )

    assert len(answers) == 3
    assert set(answers) == {"urgent", "team", "sev"}
    assert answers["team"].type == "choice"
    assert answers["team"].probabilities[1].option == "Facility"


async def test_the_async_client_matches_the_sync_one(cases):
    """The two clients share a builder, and this is what pins that they stay equivalent."""
    case = cases["responses"][0]
    body = case["body"]

    sync = client_for(200, body).systemone("state").noul("urgent", "q").send()

    async_client = AsyncClient(
        "http://cerno.test", client=httpx.AsyncClient(transport=transport(200, body))
    )
    asynchronous = await async_client.systemone("state").noul("urgent", "q").send()

    assert sync.model == asynchronous.model
    assert sync.noul("urgent") == asynchronous.noul("urgent")
    assert sync.choice("team") == asynchronous.choice("team")
    await async_client.aclose()


async def test_the_async_client_raises_the_same_errors(cases):
    case = cases["errors"][0]
    client = AsyncClient(
        "http://cerno.test",
        client=httpx.AsyncClient(transport=transport(case["status"], case["body"])),
    )

    with pytest.raises(ApiError) as caught:
        await client.systemone("state").noul("q", "Urgent?").send()

    assert caught.value.code == case["body"]["code"]
    await client.aclose()

"""Typed answers, parsed from the service's JSON.

Plain dataclasses rather than a validation library: the service already validated, and the one
dependency this client has is its HTTP transport.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Mapping


class CernoError(Exception):
    """Base for everything this package raises."""


class ApiError(CernoError):
    """The service answered with a structured failure.

    Branch on ``code``, never on ``message`` — the codes are the contract, the prose is not.
    """

    def __init__(self, status: int, code: str, message: str, question_id: str | None = None):
        super().__init__(f"cerno returned {status} ({code}): {message}")
        self.status = status
        self.code = code
        self.message = message
        self.question_id = question_id


class TransportError(CernoError):
    """The service could not be reached, or did not answer in time.

    The underlying ``httpx`` exception is kept as ``__cause__``.
    """


class UnexpectedResponse(CernoError):
    """A response that was not shaped like anything cerno sends — a proxy, most likely.

    Raised for a non-2xx body that is not a cerno error, and for a 2xx body that is not a cerno
    answer.
    """

    def __init__(self, status: int, body: str):
        super().__init__(f"cerno returned {status}: {body[:200]}")
        self.status = status
        self.body = body


class MissingAnswer(CernoError):
    def __init__(self, question_id: str):
        super().__init__(f"no answer for question {question_id!r}")
        self.question_id = question_id


class WrongAnswerType(CernoError):
    def __init__(self, question_id: str, expected: str, actual: str):
        super().__init__(f"question {question_id!r} answered with a {actual}, not a {expected}")
        self.question_id = question_id
        self.expected = expected
        self.actual = actual


@dataclass(frozen=True, slots=True)
class OptionProbability:
    option: str
    probability: float


@dataclass(frozen=True, slots=True)
class LevelProbability:
    level: int
    legend: str
    probability: float


@dataclass(frozen=True, slots=True)
class Answer:
    """One typed answer.

    ``raw_logprobs`` and ``truncated`` travel with every answer so a caller can redo the
    normalisation themselves: calibration is a convenience, never a place information is lost.
    ``truncated_labels`` names the labels whose logprob is an upper bound rather than an
    observation; a server predating it omits it, which reads as empty.
    """

    type: Literal["noul", "choice", "score"]
    confidence: float
    raw_logprobs: Mapping[str, float]
    truncated: bool
    noul: float | None = None
    choice: str | None = None
    index: int | None = None
    score: int | None = None
    expected_score: float | None = None
    legend: str | None = None
    probabilities: list[OptionProbability] | list[LevelProbability] = field(default_factory=list)
    truncated_labels: tuple[str, ...] = ()

    @classmethod
    def parse(cls, data: Mapping[str, Any]) -> "Answer":
        kind = data["type"]

        if kind == "choice":
            probabilities: Any = [
                OptionProbability(p["option"], p["probability"])
                for p in data.get("probabilities", [])
            ]
        elif kind == "score":
            probabilities = [
                LevelProbability(p["level"], p["legend"], p["probability"])
                for p in data.get("probabilities", [])
            ]
        else:
            probabilities = []

        return cls(
            type=kind,
            confidence=data["confidence"],
            raw_logprobs=dict(data.get("raw_logprobs", {})),
            truncated=data["truncated"],
            noul=data.get("noul"),
            choice=data.get("choice"),
            index=data.get("index"),
            score=data.get("score"),
            expected_score=data.get("expected_score"),
            legend=data.get("legend"),
            probabilities=probabilities,
            truncated_labels=tuple(data.get("truncated_labels", ())),
        )


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    questions: int


class Answers:
    """The answers to one request, with accessors that fail loudly on the wrong id or type."""

    __slots__ = ("_answers", "model", "usage", "timing_ms")

    def __init__(self, data: Mapping[str, Any]):
        self._answers = {k: Answer.parse(v) for k, v in data["answers"].items()}
        self.model: str = data["model"]
        self.usage = Usage(data["usage"]["input_tokens"], data["usage"]["questions"])
        self.timing_ms: int = data["timing_ms"]["total"]

    def __getitem__(self, question_id: str) -> Answer:
        try:
            return self._answers[question_id]
        except KeyError:
            raise MissingAnswer(question_id) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._answers)

    def __len__(self) -> int:
        return len(self._answers)

    def __repr__(self) -> str:
        return f"Answers(model={self.model!r}, ids={list(self._answers)})"

    def _typed(self, question_id: str, expected: str) -> Answer:
        answer = self[question_id]
        if answer.type != expected:
            raise WrongAnswerType(question_id, expected, answer.type)
        return answer

    def noul(self, question_id: str) -> float:
        """Probability that the answer is yes."""
        value = self._typed(question_id, "noul").noul
        assert value is not None  # guaranteed by the type check above
        return value

    def choice(self, question_id: str) -> str:
        """The winning option."""
        value = self._typed(question_id, "choice").choice
        assert value is not None
        return value

    def index(self, question_id: str) -> int:
        """The winning option's position in the request."""
        value = self._typed(question_id, "choice").index
        assert value is not None
        return value

    def score(self, question_id: str) -> int:
        """The winning level, 1-based."""
        value = self._typed(question_id, "score").score
        assert value is not None
        return value

    def expected_score(self, question_id: str) -> float:
        """The probability-weighted mean level."""
        value = self._typed(question_id, "score").expected_score
        assert value is not None
        return value

    def legend(self, question_id: str) -> str:
        value = self._typed(question_id, "score").legend
        assert value is not None
        return value

    def confidence(self, question_id: str) -> float:
        return self[question_id].confidence

    def truncated(self, question_id: str) -> bool:
        """Whether some label fell outside the host's reporting window, making its probability
        an upper bound rather than an observation."""
        return self[question_id].truncated

    def truncated_labels(self, question_id: str) -> tuple[str, ...]:
        """The labels whose logprob is an upper bound rather than an observation."""
        return self[question_id].truncated_labels

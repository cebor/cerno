"""Assembling a request.

The builder is transport-agnostic on purpose: it produces the request body, and the sync and
async clients differ only in how they send it. That is what keeps the two clients from drifting.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence


class SystemOneBuilder:
    """A request under construction. Chain questions onto it, then ``send()``."""

    __slots__ = ("_body", "_send")

    def __init__(self, state: str, send: Any):
        self._body: dict[str, Any] = {"state": state, "questions": []}
        self._send = send

    def model(self, model: str) -> "SystemOneBuilder":
        """Name a model or a configured alias. The service's default applies otherwise."""
        self._body["model"] = model
        return self

    def calibration(self, temperature: float) -> "SystemOneBuilder":
        """Scale the label logits before normalising. Above 1 flattens, below 1 sharpens."""
        self._body["calibration"] = {"temperature": temperature}
        return self

    def noul(self, question_id: str, question: str) -> "SystemOneBuilder":
        """How likely the answer to ``question`` is yes."""
        self._body["questions"].append({"id": question_id, "noul": question})
        return self

    def choice(
        self,
        question_id: str,
        question: str | None,
        options: Iterable[str] | None = None,
    ) -> "SystemOneBuilder":
        """One of ``options``.

        ``question`` may be ``None`` when the options speak for themselves. Passing the options
        as the second argument works too, for the same case.
        """
        if options is None:
            options, question = question, None  # type: ignore[assignment]

        # A string is iterable, so a forgotten options list would otherwise be sent as one
        # option per character — and the service would answer it.
        if isinstance(options, str) or options is None:
            raise TypeError(
                f"choice {question_id!r} needs a list of options, got {type(options).__name__}; "
                "pass the question and then the options, or the options alone"
            )

        spec: dict[str, Any] = {}
        if question is not None:
            spec["question"] = question
        spec["options"] = list(options)

        self._body["questions"].append({"id": question_id, "choice": spec})
        return self

    def score(
        self,
        question_id: str,
        question: str,
        levels: int | Sequence[str],
    ) -> "SystemOneBuilder":
        """A position on a rubric: an int for generated levels, or the level texts."""
        # bool is an int subclass, and a str is a sequence of characters; neither is a rubric.
        if isinstance(levels, bool) or isinstance(levels, str):
            raise TypeError(
                f"score {question_id!r} needs a level count or a list of level texts, "
                f"got {type(levels).__name__}"
            )

        spec: dict[str, Any] = {"question": question}
        spec["levels"] = levels if isinstance(levels, int) else list(levels)

        self._body["questions"].append({"id": question_id, "score": spec})
        return self

    def body(self) -> dict[str, Any]:
        """The request as it will be sent. Useful for logging, and for testing a chain without
        a server."""
        return self._body

    def send(self):
        """Send the request. Returns ``Answers``, or an awaitable for one on an async client."""
        return self._send(self._body)

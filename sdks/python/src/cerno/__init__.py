"""Python client for cerno.

cerno answers three kinds of question about a piece of text — is it true (``noul``), which one
is it (``choice``), where on a scale does it sit (``score``) — using a locally hosted model.
Each question is one forward pass, so answers come back in tens of milliseconds.

    from cerno import Client

    client = Client("http://localhost:3000")
    answers = (
        client.systemone("Ticket: server room at 31C, rising.")
        .noul("urgent", "Is this urgent?")
        .choice("team", "Which team?", ["IT", "Facility", "HR"])
        .score("sev", "How severe?", 5)
        .send()
    )

    answers.noul("urgent")      # 0.991
    answers.choice("team")      # "Facility"
    answers.score("sev")        # 5
"""

from ._builder import SystemOneBuilder
from ._models import (
    Answer,
    Answers,
    ApiError,
    CernoError,
    LevelProbability,
    MissingAnswer,
    OptionProbability,
    TransportError,
    UnexpectedResponse,
    Usage,
    WrongAnswerType,
)
from .client import AsyncClient, Client

__all__ = [
    "Answer",
    "Answers",
    "ApiError",
    "AsyncClient",
    "CernoError",
    "Client",
    "LevelProbability",
    "MissingAnswer",
    "OptionProbability",
    "SystemOneBuilder",
    "TransportError",
    "UnexpectedResponse",
    "Usage",
    "WrongAnswerType",
]

__version__ = "0.1.1"

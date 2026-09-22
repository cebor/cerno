import json
from pathlib import Path

import pytest

CASES = Path(__file__).resolve().parents[3] / "spec" / "conformance" / "cases.json"


@pytest.fixture(scope="session")
def cases():
    return json.loads(CASES.read_text())

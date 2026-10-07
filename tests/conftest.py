import os
import tempfile
from pathlib import Path

import pytest

from forecasting.data import generate_sales
from forecasting.train import train_pipeline

ARTIFACTS = Path(tempfile.mkdtemp(prefix="forecast-tests-"))
os.environ["ARTIFACT_DIR"] = str(ARTIFACTS)

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

HORIZON = 7


@pytest.fixture(scope="session")
def small_df():
    # 2 stores x 2 items x ~2 years keeps training to a few seconds
    return generate_sales(start="2023-01-01", end="2024-12-31", n_stores=2, n_items=2, seed=7)


@pytest.fixture(scope="session")
def metadata(small_df):
    return train_pipeline(small_df, ARTIFACTS, horizon=HORIZON, n_estimators=60, do_tune=False, n_origins=2)


@pytest.fixture(scope="session")
def client(metadata):
    with TestClient(app) as c:
        yield c

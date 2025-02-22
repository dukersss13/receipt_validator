import pytest
import os
from time import time

from src.data_reader import DataReader, DataType


data_path = "tests/data/"


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true", reason="Skipping on GitHub Actions"
)
def test_data_reader():
    dr = DataReader()

    start = time()
    image_payload = dr.create_image_payload(data_path)

    _ = dr.read_data(image_payload)
    end = time()

    t3 = end-start

    print(f"Time to read: {round(t3, 2)}s")

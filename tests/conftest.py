import pytest


@pytest.fixture(scope="session")
def ray_session():
    # ray.init()
    yield None
    # ray.shutdown()

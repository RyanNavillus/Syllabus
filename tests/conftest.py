import pytest
import ray

@pytest.fixture(scope="session")
def ray_session():
    ray.init()
    yield None
    ray.shutdown()

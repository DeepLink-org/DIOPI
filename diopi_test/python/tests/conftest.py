import pytest


def pytest_addoption(parser):
    parser.addoption("--test-all-models", action="store_true",
                     help="test all models or one model for each category")


@pytest.fixture(scope="session")
def test_all(request):
    return request.config.getoption("--test-all-models")

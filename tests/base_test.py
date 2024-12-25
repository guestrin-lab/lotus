import pytest


class BaseTest:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Set up any common configurations or fixtures
        yield
        # Teardown (if needed)

    # Add any common utility methods here

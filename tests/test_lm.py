from lotus.models import LM
from tests.base_test import BaseTest


class TestLM(BaseTest):
    def test_lm_initialization(self):
        lm = LM(model="gpt-4o-mini")
        assert isinstance(lm, LM)

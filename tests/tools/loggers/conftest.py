from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_wandb():
    wandb_mock = MagicMock()
    with patch.dict("sys.modules", {"wandb": wandb_mock}):
        yield wandb_mock

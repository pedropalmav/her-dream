from unittest.mock import patch

import pytest


@pytest.fixture
def mock_summary_writer_logging():
    with patch("her_dream.tools.logging.SummaryWriter") as m:
        yield m


@pytest.fixture
def mock_summary_writer_tb():
    with patch("her_dream.tools.loggers.tensorboard_logger.SummaryWriter") as m:
        yield m


@pytest.fixture
def mock_write_video_logging():
    with patch("her_dream.tools.logging.torchvision.io.write_video") as m:
        yield m


@pytest.fixture
def mock_write_video_file():
    with patch("her_dream.tools.loggers.file_logger.torchvision.io.write_video") as m:
        yield m

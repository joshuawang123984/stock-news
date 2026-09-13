"""Tests for local_generation.py"""

import pytest
from unittest.mock import patch, MagicMock
from local_generation import generate_summary
from resources import get_gpu_layers

# --- get_gpu_layers ---

@patch("resources.torch")
def test_get_gpu_layers_returns_negative_one_when_cuda_available(mock_torch):
    """Should offload all layers when a gpu is detected."""
    mock_torch.cuda.is_available.return_value = True
    assert get_gpu_layers() == -1


@patch("resources.torch")
@patch("resources.platform")
def test_get_gpu_layers_returns_zero_when_no_gpu_detected(mock_platform, mock_torch):
    """Should fall back to cpu when no gpu is found."""
    mock_torch.cuda.is_available.return_value = False
    mock_platform.system.return_value = "Linux"  
    assert get_gpu_layers() == 0


# --- generate_summary ---

@patch("local_generation.get_llm")
def test_generate_summary_includes_articles_in_prompt(mock_get_llm):
    """The prompt sent to the model should include each article's title
    and description."""
    mock_model = MagicMock()
    mock_model.return_value = {"choices": [{"text": "This is a test answer."}], "usage": {"prompt_tokens": 50, "completion_tokens": 10}}
    mock_get_llm.return_value = mock_model

    articles = [
        {"title": "SRPT reports earnings", "description": "Strong quarter."},
    ]

    result, stats = generate_summary("What's new on SRPT?", articles)

    called_prompt = mock_model.call_args[0][0]
    assert "SRPT reports earnings" in called_prompt
    assert "Strong quarter." in called_prompt
    assert result == "This is a test answer."


@patch("local_generation.get_llm")
def test_generate_summary_handles_no_articles(mock_get_llm):
    """Should not crash if given an empty article list.
    the model should report it can't answer."""
    mock_model = MagicMock()
    mock_model.return_value = {"choices": [{"text": "No information available."}], "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
    mock_get_llm.return_value = mock_model

    result, stats = generate_summary("What's new on SRPT?", [])

    assert result == "No information available."
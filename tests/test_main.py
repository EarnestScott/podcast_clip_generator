import json
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, '.')
from main import find_clip_for_topic


def _mock_openai(content: str):
    """Helper: returns a mock OpenAI client whose chat completion returns `content`."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = content
    return mock_client


def test_find_clip_high_confidence():
    payload = {
        "start_time": 142.5,
        "end_time": 198.0,
        "title": "Vitamin D3 and mood regulation",
        "clip_text": "Vitamin D3 has significant impacts on mood...",
        "confidence": "high"
    }
    with patch('main.openai.OpenAI', return_value=_mock_openai(json.dumps(payload))):
        result = find_clip_for_topic("...transcript...", "vitamin d3 and mood")

    assert result["confidence"] == "high"
    assert result["start_time"] == 142.5
    assert result["end_time"] == 198.0
    assert result["title"] == "Vitamin D3 and mood regulation"


def test_find_clip_not_found():
    payload = {
        "start_time": 0,
        "end_time": 0,
        "title": "",
        "clip_text": "",
        "confidence": "not_found"
    }
    with patch('main.openai.OpenAI', return_value=_mock_openai(json.dumps(payload))):
        result = find_clip_for_topic("...transcript...", "aliens on mars")

    assert result["confidence"] == "not_found"


def test_find_clip_strips_markdown_fences():
    payload = {
        "start_time": 50.0,
        "end_time": 100.0,
        "title": "Test clip",
        "clip_text": "Some text",
        "confidence": "high"
    }
    wrapped = "```json\n" + json.dumps(payload) + "\n```"
    with patch('main.openai.OpenAI', return_value=_mock_openai(wrapped)):
        result = find_clip_for_topic("...transcript...", "test topic")

    assert result["confidence"] == "high"
    assert result["start_time"] == 50.0

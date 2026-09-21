"""Test module for cli Typer commands.

This module contains test cases for the CLI commands in the
osc_transformer_based_extractor.cli module.
"""

import pytest
from typer.testing import CliRunner

from osc_transformer_based_extractor.main_cli import (
    app,
)  # Adjust the import based on your package structure

runner = CliRunner()


@pytest.fixture
def mock_relevance_detector_app(mocker):
    return mocker.patch(
        "my_package.relevance_detector.cli_relevance_detector.relevance_detector_app"
    )


@pytest.fixture
def mock_question_answering_app(mocker):
    return mocker.patch("my_package.question_answering.cli_qna.question_answering_app")


def test_main_no_command(mock_relevance_detector_app, mock_question_answering_app):
    result = runner.invoke(app)
    assert result.exit_code == 0
    assert "OSC Transformer Based Extractor CLI." in result.output
    assert "Available commands:" in result.output
    assert "- relevance-detector" in result.output
    assert "- question-answering" in result.output


def test_relevance_detector_command(mock_relevance_detector_app):
    result = runner.invoke(app, ["relevance-detector"])
    assert result.exit_code == 0
    mock_relevance_detector_app.assert_called_once()


def test_question_answering_command(mock_question_answering_app):
    result = runner.invoke(app, ["question-answering"])
    assert result.exit_code == 0
    mock_question_answering_app.assert_called_once()

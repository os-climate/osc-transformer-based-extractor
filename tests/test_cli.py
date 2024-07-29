"""Test module for cli Typer commands.

This module contains test cases for the CLI commands in the
osc_transformer_based_extractor.cli module.
"""

from typer.testing import CliRunner
from unittest.mock import patch
from osc_transformer_based_extractor.cli import (
    app,
)  # Import your Typer application instance

runner = CliRunner()


# Basic fine_tune command test
@patch("osc_transformer_based_extractor.fine_tune.fine_tune_model")
@patch("osc_transformer_based_extractor.fine_tune.check_csv_columns")
@patch("osc_transformer_based_extractor.fine_tune.check_output_dir")
def test_fine_tune_command(
    mock_check_output_dir, mock_check_csv_columns, mock_fine_tune_model
):
    """Test the fine_tune command.

    This test verifies that the fine_tune command runs without errors and calls
    the appropriate functions.
    """
    mock_fine_tune_model.return_value = None  # Mock return value if needed
    mock_check_csv_columns.return_value = None  # Mock return value if needed
    mock_check_output_dir.return_value = None  # Mock return value if needed

    result = runner.invoke(
        app,
        [
            "fine_tune",
            "tests/data.csv",
            "bert-base-uncased",
            "5",
            "128",
            "3",
            "32",
            "tests/trained_models/",
            "500",
        ],
    )

    print(result.output)  # Debug output

    assert result.exit_code == 0  # Ensure the command exits without errors
    assert "Model 'bert-base-uncased' trained and saved successfully" in result.output
    assert mock_fine_tune_model.called  # Ensure that fine_tune_model was called


# Basic perform_inference command test
@patch("osc_transformer_based_extractor.inference.get_inference")
@patch("osc_transformer_based_extractor.inference.check_question_context")
@patch("osc_transformer_based_extractor.inference.check_model_and_tokenizer_path")
def test_perform_inference_command(
    mock_check_model_and_tokenizer_path, mock_check_question_context, mock_get_inference
):
    """Test the perform_inference command.

    This test verifies that the perform_inference command runs without errors and calls
    the appropriate functions.
    """
    mock_get_inference.return_value = "Mocked Label ID"  # Mock return value if needed
    mock_check_question_context.return_value = None  # Mock return value if needed
    mock_check_model_and_tokenizer_path.return_value = (
        None  # Mock return value if needed
    )

    result = runner.invoke(
        app,
        [
            "perform_inference",
            "What is the main idea?",
            "This is the context.",
            "tests/trained_model/",
            "tests/tokenizer/",
        ],
    )

    print(result.output)  # Debug output

    assert result.exit_code == 0  # Ensure the command exits without errors
    assert "Predicted Label ID: Mocked Label ID" in result.output
    assert mock_get_inference.called  # Ensure that get_inference was called


# Testing ValueError in perform_inference command
@patch("osc_transformer_based_extractor.inference.get_inference")
@patch("osc_transformer_based_extractor.inference.check_question_context")
@patch("osc_transformer_based_extractor.inference.check_model_and_tokenizer_path")
def test_perform_inference_command_value_error(
    mock_check_model_and_tokenizer_path, mock_check_question_context, mock_get_inference
):
    """Test the perform_inference command for handling ValueError.

    This test verifies that the perform_inference command handles ValueError correctly.
    """
    mock_get_inference.side_effect = ValueError("Mocked ValueError")  # Mock side effect
    mock_check_question_context.return_value = None  # Mock return value if needed
    mock_check_model_and_tokenizer_path.return_value = (
        None  # Mock return value if needed
    )

    result = runner.invoke(
        app,
        [
            "perform_inference",
            "What is the main idea?",
            "",  # Passing empty context to raise ValueError
            "tests/trained_model/",
            "tests/tokenizer/",
        ],
    )

    print(result.output)  # Debug output

    assert result.exit_code != 0  # Expecting a non-zero exit code for ValueError
    assert "Error: Mocked ValueError" in result.output
    assert mock_get_inference.called  # Ensure that get_inference was called

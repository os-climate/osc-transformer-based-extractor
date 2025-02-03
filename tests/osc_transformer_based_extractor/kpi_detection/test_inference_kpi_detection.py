"""Test File for inference_kpi_detection"""

from unittest.mock import patch, MagicMock
from tempfile import TemporaryDirectory
from pathlib import Path
import shutil
import pandas as pd
import pytest

# Import the functions from the module
from osc_transformer_based_extractor.kpi_detection.inference_kpi_detection import (
    run_full_inference_kpi_detection,
)


@pytest.fixture
def temp_dir():
    """
    Create a temporary directory for the test.
    """
    with TemporaryDirectory() as temp_dir:
        yield temp_dir
        # Clean up the temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def create_test_data(file_path):
    """
    Create a temporary CSV file with test data for questions and contexts.
    """
    data = {
        "question": [
            "What is the capital of France?",
            "Who wrote 'To Kill a Mockingbird'?",
        ],
        "context": [
            "The capital of France is Paris.",
            "'To Kill a Mockingbird' was written by Harper Lee.",
        ],
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)


@patch(
    "src.osc_transformer_based_extractor.kpi_detection.inference_kpi_detection.pipeline"
)
@patch("pandas.read_csv")
@patch("pandas.DataFrame.to_excel")
def test_run_full_inference_kpi_detection(
    mock_to_excel, mock_read_csv, mock_pipeline, temp_dir
):
    """
    Test the run_full_inference_kpi_detection function.
    """
    # Create test input data
    data_file_path = Path(temp_dir) / "test_data.csv"
    create_test_data(data_file_path)

    # Mock the read_csv function to return the test data without reading a file
    test_data = pd.DataFrame(
        {
            "question": [
                "What is the capital of France?",
                "Who wrote 'To Kill a Mockingbird'?",
            ],
            "context": [
                "The capital of France is Paris.",
                "'To Kill a Mockingbird' was written by Harper Lee.",
            ],
        }
    )
    mock_read_csv.return_value = test_data

    # Mock the Hugging Face pipeline
    mock_qa_pipeline = MagicMock()
    mock_pipeline.return_value = mock_qa_pipeline
    mock_qa_pipeline.return_value = [
        {"answer": "Paris", "start": 24, "end": 29, "score": 0.98},
        {"answer": "Harper Lee", "start": 29, "end": 39, "score": 0.99},
    ]

    # Set up paths
    output_path = Path(temp_dir)
    model_path = "distilbert-base-uncased-distilled-squad"  # Using a well-known model for testing

    # Run the inference
    run_full_inference_kpi_detection(data_file_path, output_path, model_path)

    # Check if output file is created
    output_file = output_path / "output.xlsx"

    # Validate the content written to the output file
    expected_output_data = pd.DataFrame(
        {
            "question": [
                "What is the capital of France?",
                "Who wrote 'To Kill a Mockingbird'?",
            ],
            "context": [
                "The capital of France is Paris.",
                "'To Kill a Mockingbird' was written by Harper Lee.",
            ],
            "answer": ["Paris", "Harper Lee"],
            "start": [24, 29],
            "end": [29, 39],
            "score": [0.98, 0.99],
        }
    )

    # Mocked `to_excel` should be called once with the expected data and output file path
    mock_to_excel.assert_called_once()
    pd.testing.assert_frame_equal(mock_to_excel.call_args[0][0], expected_output_data)
    assert mock_to_excel.call_args[0][1] == output_file, (
        "Output file path is incorrect."
    )
